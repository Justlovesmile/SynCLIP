import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.misc import is_main_process
from training.distributed import is_master

import swanlab
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
import cv2
from open_clip.tokenizer import SimpleTokenizer


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


class SynCLIP:
    def __init__(self, args):
        if args.use_mlp and args.dataset_type in ['semantic_grid_distill', 'semantic_proposals_distill']:
            self.projector = build_mlp(args.mlp_hidden_size, args.mlp_proj_dim, args.mlp_zdim).to(args.device)
            self.initialize_weights()
        self.tokenizer = SimpleTokenizer()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def __call__(self, batch, student, teacher, vfm_model, args):
        losses={}
        context_weight = args.loss_context_weight
        content_weight = args.loss_content_weight
        semantic_weight = args.loss_semantic_weight
        if args.distributed:
            student = student.module
        dtype_map = {"bf16": torch.bfloat16, "amp": torch.float16}
        input_dtype = dtype_map.get(args.precision, torch.float32)
        if args.dataset_type in ['semantic_grid_distill', 'semantic_proposals_distill']:
            images, normed_boxes, image_crops, proxy_image, image_labels = batch
            image_labels = image_labels.to(device=args.device, dtype=torch.int64, non_blocking=True)
        else:
            images, normed_boxes, image_crops, proxy_image = batch
        images = images.to(device=args.device, dtype=input_dtype, non_blocking=True)  
        normed_boxes = normed_boxes.to(device=args.device, dtype=input_dtype, non_blocking=True)
        image_crops = image_crops.to(device=args.device, dtype=input_dtype, non_blocking=True) 
        proxy_image = proxy_image.to(device=args.device, dtype=input_dtype, non_blocking=True)

        rois_list = []
        crops_list = []
        for bboxes_per_image, crops_per_image in zip(normed_boxes, image_crops):
            valid = bboxes_per_image[:, -1] > 0.5
            rois_list.append(bboxes_per_image[valid, :4])
            crops_list.append(crops_per_image[valid])
        image_crops = torch.cat(crops_list)
        student_roi_features, context = student.encode_pseudo_boxes(images, rois_list, normalize=True, mode=args.mode)

        with torch.no_grad():
            teacher_crop_features = teacher.encode_image(image_crops, normalize=True)
            if args.use_vfm:
                teacher_context_similarity, teacher_h, teacher_w = self.get_teacher_context_similarity(vfm_model, proxy_image, args)
            else:
                teacher_context_similarity = teacher_h = teacher_w = None

        if args.use_vfm:
            student_context_similarity = self.get_student_context_similarity(images, context, teacher_h, teacher_w, args)
            _loss_context = (teacher_context_similarity - student_context_similarity).norm(p=2, dim=-1).mean() 
            losses.update({"loss_context": _loss_context * context_weight})

        if args.dataset_type in ['semantic_grid_distill', 'semantic_proposals_distill']:
            _loss_semantic = self.get_semantic_similarity_loss(teacher, student, images, image_labels, input_dtype, teacher_context_similarity, args)
            losses.update({"loss_semantic": _loss_semantic * semantic_weight})

        _loss_content = 1.0 - (student_roi_features * teacher_crop_features).sum(-1).mean()
        losses.update({"loss_content": _loss_content * content_weight})
        return losses, len(images)

    def get_teacher_context_similarity(self, vfm_model, proxy_image, args):
        if "sam" in args.use_vfm:
            vfm_feats = vfm_model.image_encoder(proxy_image)
        elif "dinov3" in args.use_vfm:
            vfm_feats = vfm_model.get_intermediate_layers(proxy_image, reshape=True)[0]
        elif "dinov2" in args.use_vfm:
            vfm_feats = vfm_model.get_intermediate_layers(proxy_image, reshape=True)[0]
        elif "dino" in args.use_vfm:
            feat = vfm_model.get_intermediate_layers(proxy_image)[0]
            nb_im = feat.shape[0]
            patch_size = vfm_model.patch_embed.patch_size
            I, J = proxy_image[0].shape[-2] // patch_size, proxy_image[0].shape[-2] // patch_size
            vfm_feats = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
        else:
            raise NotImplementedError(f"mode {args.use_vfm} is not implemented yet.")
        teacher_h, teacher_w = vfm_feats.shape[-2:]
        vfm_feats = F.normalize(vfm_feats.flatten(-2,-1), dim=1)
        vfm_similarity = torch.einsum("b c m, b c n -> b m n", vfm_feats, vfm_feats)
        return vfm_similarity, teacher_h, teacher_w
    
    def get_student_context_similarity(self, images, context, teacher_h, teacher_w, args):
        B = images.shape[0]
        if args.mode in ["qq_vfm_distill", "kk_vfm_distill", "vv_vfm_distill", "sanity_check"]:
            N, _ = context.shape[1:]
            context = context.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
            bs, N, C = context.shape
            n_sqrt = int(N ** 0.5)
            if n_sqrt != teacher_h or n_sqrt != teacher_w:
                context_reshaped = context.transpose(-2,-1).contiguous().view(bs, C, n_sqrt, n_sqrt)
                context_resized = F.interpolate(context_reshaped, size=(teacher_h, teacher_w), mode='bilinear', align_corners=False)
                context = context_resized.transpose(-2,-1).contiguous().view(bs, teacher_h * teacher_w, C)
            context = F.normalize(context, dim=-1).transpose(-2,-1)
            student_context_similarity=torch.einsum("b c m, b c n -> b m n", context, context)
        elif args.mode == "csa_vfm_distill":
            q_feature, k_feature = context
            N, _ = q_feature.shape[1:]
            q_feature = q_feature.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
            k_feature = k_feature.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
            q_feature = F.normalize(q_feature, dim=-1).transpose(-2,-1)
            k_feature = F.normalize(k_feature, dim=-1).transpose(-2,-1)
            student_context_similarity = (torch.einsum("b c m, b c n -> b m n", q_feature, q_feature) + 
                                          torch.einsum("b c m, b c n -> b m n", k_feature, k_feature)) / 2.0
        elif args.mode == "all_vfm_distill":
            q_feature, k_feature, v_feature = context
            q_feature = F.normalize(q_feature, dim=-1).transpose(-2,-1)
            k_feature = F.normalize(k_feature, dim=-1).transpose(-2,-1)
            v_feature = F.normalize(v_feature, dim=-1).transpose(-2,-1)
            student_context_similarity = (torch.einsum("b c m, b c n -> b m n", q_feature, q_feature)+
                                          torch.einsum("b c m, b c n -> b m n", k_feature, k_feature)+
                                          torch.einsum("b c m, b c n -> b m n", v_feature, v_feature)) / 3.0
        else:
            raise NotImplementedError(f"Mode '{args.mode}' is not implemented.")
        return student_context_similarity
    
    def get_semantic_similarity_loss(self, teacher, student, images, image_labels, input_dtype, vfm_similarity, args):
        """
        images: b, c, h, w
        image_labels: b, max_semantic*2, max_tokens+1
        """
        # Step 1: Split into label tokens and semantic tokens
        label_tokens = image_labels[:, :args.max_semantic, :args.max_tokens] # b, max_semantic, max_tokens
        semantic_tokens = image_labels[:, args.max_semantic:, :args.max_tokens] # b, max_semantic, max_tokens

        # Step 2: Extract vaild mask
        label_valid = image_labels[:, :args.max_semantic, -1] > 0.5 # b, max_semantic
        semantic_valid = image_labels[:, args.max_semantic:, -1] > 0.5 # b, max_semantic
        assert torch.all(label_valid == semantic_valid), "Mismatch in label/semantic valid mask"
        valid_mask = label_valid

        # Step 3: Truncate tokens and flatten to 2D
        label_flat = label_tokens.reshape(-1, args.max_tokens) # b*max_semantic, max_tokens
        semantic_flat = semantic_tokens.reshape(-1, args.max_tokens) # b*max_semantic, max_tokens
        valid_flat = valid_mask.reshape(-1) # # b*max_semantic

        # Step 4: Encode valid text features
        with torch.no_grad():
            label_feats_all = teacher.encode_text(label_flat[valid_flat]).to(dtype=input_dtype) # bn_valid, D
            semantic_feats_all = teacher.encode_text(semantic_flat[valid_flat]).to(dtype=input_dtype) # bn_valid, D

        # Step 5: Group back to (B, N, D)
        B = image_labels.shape[0]
        D = label_feats_all.shape[-1]
        valid_counts = valid_mask.sum(dim=1).tolist() # len=b, sum=bn_valid

        def pad_and_mask(feats):
            N_max = max(valid_counts)
            padded = torch.zeros(B, N_max, D, device=feats.device)
            mask = torch.zeros(B, N_max, dtype=torch.bool, device=feats.device)
            offset = 0
            for i, count in enumerate(valid_counts):
                if count == 0:
                    continue
                padded[i, :count] = feats[offset:offset+count]
                mask[i, :count] = True
                offset += count
            return padded, mask

        label_feats, valid_mask_padded = pad_and_mask(label_feats_all)           # (B, N_max, D), (B, N_max)
        semantic_feats, _ = pad_and_mask(semantic_feats_all)

        # Step 6: Normalize text features
        need_normalize = not args.no_normal
        if need_normalize:
            label_feats = F.normalize(label_feats, dim=2)      # (B, N_max, D)
            semantic_feats = F.normalize(semantic_feats, dim=2)

        # Step 7: Get image feats
        student_image_feats = student.encode_dense(images, normalize=need_normalize, mode=args.sem_mode) # B, M, D
        teacher_image_feats = teacher.encode_dense(images, normalize=need_normalize, mode=args.sem_mode) # B, M, D
        # image_feats = student.encode_image(images, normalize=True) # B, D

        # additional: use MLP
        if args.use_mlp:
            B, M, D = student_image_feats.shape
            student_image_feats = self.projector(student_image_feats.reshape(-1, D)).reshape(B, M, -1)

        # Step 8: Compute Similarity
        sim_label = torch.einsum('b n d, b m d -> b n m', label_feats, student_image_feats)
        # sim_label = torch.softmax(sim_label/math.sqrt(D), dim=-1)
        sim_semantic = torch.einsum('b n d, b m d -> b n m', semantic_feats, teacher_image_feats)
        # sim_semantic = torch.softmax(sim_semantic/math.sqrt(D), dim=-1)

        # additional: use vfm
        if args.use_topk and args.use_vfm:
            B, N, M = sim_semantic.shape
            if args.filter_black:
                # TODO: mask invalid region for better topk selection
                B, M, D = student_image_feats.shape
                H = W = int(M**0.5)
                images_resized = F.interpolate(self.denormalize(images.detach()), size=(H, W), mode='bilinear')
                images_resized = images_resized.mean(dim=1)            # B, C, H, W -> B, H, W
                mins = images_resized.view(B, -1).min(dim=1)[0]   # [B]
                img_thr = mins.view(B, 1, 1)
                valid_token_mask = (images_resized > img_thr).float().view(B, 1, M)
                _, topk_indices = torch.topk(sim_semantic * valid_token_mask, k=args.topk, dim=-1) # B, N, K
                valid_token_mask = valid_token_mask + 0.2 * (1 - valid_token_mask)
                sim_semantic = sim_semantic * valid_token_mask
            else:
                _, topk_indices = torch.topk(sim_semantic, k=args.topk, dim=-1)      # B, N, K
            vfm_sim_mat = vfm_similarity.unsqueeze(1).expand(-1, N, -1, -1)          # B, N, M, M
            topk_idx_mat = topk_indices.unsqueeze(-1).expand(-1, -1, -1, M)          # B, N, K, M
            selected_vfm_sim = torch.gather(vfm_sim_mat, dim=2, index=topk_idx_mat)  # B, N, K, M
            sim_vfm_mean = selected_vfm_sim.mean(dim=2)                              # B, N, M
            # sim_vfm_mean = torch.softmax(sim_vfm_mean/math.sqrt(D), dim=-1)
            if args.use_reverse_topk:
                _, min_topk_indices = torch.topk(sim_semantic, k=args.reverse_topk, dim=-1, largest=False) # B, N, K
                min_topk_idx_mat = min_topk_indices.unsqueeze(-1).expand(-1, -1, -1, M)         # B, N, K, M
                min_selected_vfm_sim = torch.gather(vfm_sim_mat, dim=2, index=min_topk_idx_mat) # B, N, K, M
                min_sim_vfm_mean = min_selected_vfm_sim.mean(dim=2)                             # B, N, M
                sim_semantic_aug = args.sem_ratio * sim_semantic + args.spa_ratio * (sim_vfm_mean - min_sim_vfm_mean)
            else:
                sim_semantic_aug = args.sem_ratio * sim_semantic + args.spa_ratio * sim_vfm_mean
        else:
            sim_semantic_aug = sim_semantic
            sim_vfm_mean = None

        # Step 9: Mask invalid entries
        sim_label = sim_label.masked_fill(~valid_mask_padded.unsqueeze(-1), 0)
        sim_semantic_aug = sim_semantic_aug.masked_fill(~valid_mask_padded.unsqueeze(-1), 0)

        # Step 10: Compute alignment loss
        if args.sem_loss == "mse":
            loss = F.mse_loss(sim_label, sim_semantic_aug, reduction='none') # (B, N)
            loss = loss.masked_fill(~valid_mask_padded.unsqueeze(-1), 0)
            loss = loss.sum() / valid_mask_padded.sum().clamp(min=1)         # avoid divide by zero
        elif args.sem_loss == "l2":
            loss = (sim_semantic_aug - sim_label).norm(p=2, dim=-1)
            loss = loss.masked_fill(~valid_mask_padded, 0)
            loss = loss.sum() / valid_mask_padded.sum().clamp(min=1)
        elif args.sem_loss == "huber":
            loss = F.huber_loss(sim_label, sim_semantic_aug, reduction='none', delta=0.1)
            loss = loss.masked_fill(~valid_mask_padded.unsqueeze(-1), 0)
            loss = loss.sum() / valid_mask_padded.sum().clamp(min=1)
        elif args.sem_loss == "huber_kl":
            hb_loss = F.huber_loss(sim_label, sim_semantic_aug, reduction='none', delta=0.1)
            hb_loss = hb_loss.masked_fill(~valid_mask_padded.unsqueeze(-1), 0)
            hb_loss = hb_loss.sum() / valid_mask_padded.sum().clamp(min=1)
            p = F.softmax(sim_label, dim=-1)
            q = F.softmax(sim_semantic_aug, dim=-1)
            kl = F.kl_div(q.log(), p, reduction='none').sum(dim=-1)          # (B, N)
            kl = kl.masked_fill(~valid_mask_padded, 0)
            kl_loss = kl.sum() / valid_mask_padded.sum().clamp(min=1)
            loss = hb_loss + 0.1 * kl_loss
        elif args.sem_loss == "mse_kl":
            mse_loss = F.mse_loss(sim_label, sim_semantic_aug, reduction='none')
            mse_loss = mse_loss.masked_fill(~valid_mask_padded.unsqueeze(-1), 0)
            mse_loss = mse_loss.sum() / valid_mask_padded.sum().clamp(min=1)
            p = F.softmax(sim_label, dim=-1)
            q = F.softmax(sim_semantic_aug, dim=-1)
            kl = F.kl_div(q.log(), p, reduction='none').sum(dim=-1)          # (B, N)
            kl = kl.masked_fill(~valid_mask_padded, 0)
            kl_loss = kl.sum() / valid_mask_padded.sum().clamp(min=1)
            loss = mse_loss + 0.1 * kl_loss
        elif args.sem_loss == "huber_cos":
            hb_loss = F.huber_loss(sim_label, sim_semantic_aug, reduction='none', delta=0.1)
            hb_loss = hb_loss.masked_fill(~valid_mask_padded.unsqueeze(-1), 0)
            hb_loss = hb_loss.sum() / valid_mask_padded.sum().clamp(min=1)
            p = F.softmax(sim_label, dim=-1)
            q = F.softmax(sim_semantic_aug, dim=-1)
            cos_sim = F.cosine_similarity(p, q, dim=-1)                      # (B, N)
            cos_loss = 1 - cos_sim
            cos_loss = cos_loss.masked_fill(~valid_mask_padded, 0)
            cos_loss = cos_loss.sum() / valid_mask_padded.sum().clamp(min=1)
            loss = hb_loss + 0.1 * cos_loss
        elif args.sem_loss == "mse_cos":
            mse_loss = F.mse_loss(sim_label, sim_semantic_aug, reduction='none')
            mse_loss = mse_loss.masked_fill(~valid_mask_padded.unsqueeze(-1), 0)
            mse_loss = mse_loss.sum() / valid_mask_padded.sum().clamp(min=1)
            p = F.softmax(sim_label, dim=-1)
            q = F.softmax(sim_semantic_aug, dim=-1)
            cos_sim = F.cosine_similarity(p, q, dim=-1)                      # (B, N)
            cos_loss = 1 - cos_sim
            cos_loss = cos_loss.masked_fill(~valid_mask_padded, 0)
            cos_loss = cos_loss.sum() / valid_mask_padded.sum().clamp(min=1)
            loss = mse_loss + 0.1 * cos_loss
        elif args.sem_loss == "focalmse":
            weight = sim_semantic_aug.softmax(dim=-1)
            loss = ((sim_semantic_aug - sim_label) ** 2) * weight
            loss = loss.masked_fill(~valid_mask_padded.unsqueeze(-1), 0)
            loss = loss.sum() / valid_mask_padded.sum().clamp(min=1)
        elif args.sem_loss == "kl":
            p = F.softmax(sim_label, dim=-1)
            q = F.softmax(sim_semantic_aug, dim=-1)
            kl = F.kl_div(q.log(), p, reduction='none').sum(dim=-1)          # (B, N)
            kl = kl.masked_fill(~valid_mask_padded, 0)
            loss = kl.sum() / valid_mask_padded.sum().clamp(min=1)
        elif args.sem_loss == "js":
            p = F.softmax(sim_label, dim=-1)
            q = F.softmax(sim_semantic_aug, dim=-1)
            m = 0.5 * (p + q)
            js = 0.5 * (F.kl_div(p.log(), m, reduction='none').sum(dim=-1) +
                        F.kl_div(q.log(), m, reduction='none').sum(dim=-1))
            js = js.masked_fill(~valid_mask_padded, 0)
            loss = js.sum() / valid_mask_padded.sum().clamp(min=1)
        elif args.sem_loss == "cos":
            p = F.softmax(sim_label, dim=-1)
            q = F.softmax(sim_semantic_aug, dim=-1)
            cos_sim = F.cosine_similarity(p, q, dim=-1)                      # (B, N)
            cos_loss = 1 - cos_sim
            cos_loss = cos_loss.masked_fill(~valid_mask_padded, 0)
            loss = cos_loss.sum() / valid_mask_padded.sum().clamp(min=1)
        else:
            raise NotImplementedError

        # ====================== Visualization ======================
        step = args.step
        if is_master(args) and step is not None and step % args.viz_interval == 0:
            # only visualize the first image in the batch for simplicity
            img_idx = 0
            # get original image for visualization
            original_img = self.denormalize(images[img_idx].detach().cpu()).numpy()
            # get similarity matrices and valid texts for visualization
            sim_label_np = sim_label[img_idx].detach().cpu().numpy()
            sim_semantic_np = sim_semantic[img_idx].detach().cpu().numpy()
            if sim_vfm_mean is not None:
                sim_vfm_mean_np = sim_vfm_mean[img_idx].detach().cpu().numpy()
            else:
                sim_vfm_mean_np = None
            
            # get valid texts for visualization
            valid_texts = []
            for i in range(label_tokens.shape[1]):
                if valid_mask[img_idx, i]:
                    # transform token IDs back to text
                    token_ids = label_tokens[img_idx, i].detach().cpu().numpy()
                    # text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                    text = ''
                    for t in token_ids:
                        if t not in [49406, 49407]:
                            text += self.tokenizer.decoder.get(t, '')
                    valid_texts.append(text.replace('</w>', ' '))

            valid_semantics = []
            for i in range(semantic_tokens.shape[1]):
                if valid_mask[img_idx, i]:
                    # transform token IDs back to text
                    token_ids = semantic_tokens[img_idx, i].detach().cpu().numpy()
                    # text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                    text = ''
                    for t in token_ids:
                        if t not in [49406, 49407]:
                            text += self.tokenizer.decoder.get(t, '')
                    valid_semantics.append(text.replace('</w>', ' '))
            
            # visualize similarity matrices and overlays
            self.viz_similarity(
                original_img=original_img,
                sim_label=sim_label_np,
                sim_semantic=sim_semantic_np,
                sim_vfm=sim_vfm_mean_np,
                valid_texts=valid_texts,
                valid_semantics=valid_semantics,
                step=step,
                viz_interval=args.viz_interval,
                args=args,
            )
        
        return loss

    def viz_similarity(self, original_img, sim_label, sim_semantic, sim_vfm, valid_texts, valid_semantics, step, viz_interval, args):
        """
        visualize similarity matrices and overlay them on the original image
        """
        # ensure it's the right step for visualization
        if step % viz_interval != 0:
            return
        
        # create visualization canvas
        fig, axes = plt.subplots(2, 4, figsize=(18, 12))
        fig.suptitle(f'Step {step}: Semantic Similarity Visualization', fontsize=16)
        
        # show original image
        self.plot_image(axes[0, 0], original_img, "Original Image")
        
        # show student similarity heatmap
        self.plot_similarity_heatmap(
            axes[0, 1], 
            sim_label[:len(valid_texts), :], 
            valid_texts, 
            "Student Similarity Matrix"
        )
        
        # show teacher similarity heatmap
        self.plot_similarity_heatmap(
            axes[0, 2], 
            sim_semantic[:len(valid_texts), :], 
            valid_semantics, 
            "Teacher Similarity Matrix"
        )
        
        # overlay student similarity on image
        self.plot_overlay(
            axes[1, 0], 
            original_img, 
            sim_label[:len(valid_texts), :], 
            "Student Similarity Overlay"
        )
        
        # overlay teacher similarity on image
        self.plot_overlay(
            axes[1, 1], 
            original_img, 
            sim_semantic[:len(valid_texts), :], 
            "Teacher Similarity Overlay"
        )
        
        if not args.use_topk:
            # difference visualization
            diff = np.abs(sim_label - sim_semantic)
            self.plot_overlay(
                axes[1, 2], 
                original_img, 
                diff[:len(valid_texts), :], 
                "Similarity Difference"
            )
        else:
            diff = np.abs(sim_label - sim_semantic)
            self.plot_similarity_heatmap(
                axes[0, 3], 
                sim_vfm[:len(valid_texts), :], 
                valid_semantics, 
                "VFM Similarity Matrix"
            )
            self.plot_overlay(
                axes[1, 2], 
                original_img, 
                sim_vfm[:len(valid_texts), :], 
                "VFM Similarity Overlay"
            )
            self.plot_overlay(
                axes[1, 3], 
                original_img, 
                args.sem_ratio * sim_semantic[:len(valid_texts), :] + args.spa_ratio * sim_vfm[:len(valid_texts), :], 
                "VFM + Teacher Similarity Overlay"
            )
        
        # save to SwanLab
        self.log_to_swanlab(fig, step, valid_texts, valid_semantics, args)
        plt.close(fig)

    def plot_image(self, ax, img, title):
        """show original image"""
        if len(img.shape) == 3 and img.shape[0] == 3:  # [C, H, W]
            img = np.transpose(img, (1, 2, 0))
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    def plot_similarity_heatmap(self, ax, sim_matrix, labels, title):
        """show similarity heatmap"""
        # only show valid part of the similarity matrix
        label_idx = 0
        # num_valid = len(labels)
        grid_size = int(np.sqrt(sim_matrix.shape[1]))
        sim_matrix = sim_matrix[label_idx, :].reshape(grid_size, grid_size)
        
        im = ax.imshow(sim_matrix, cmap='viridis', aspect='auto')
        
        # set labels
        ax.set_title(title)
        if len(labels) > 0:
            ax.set_xlabel(f'{labels[label_idx][:40]}')
        else:
            ax.set_xlabel('')
        ax.set_ylabel('')

    def plot_overlay(self, ax, original_img, sim_matrix, title):
        """overlay similarity on the original image"""
        label_idx = 0
        # get the number of spatial positions
        num_positions = sim_matrix.shape[1]
        grid_size = int(np.sqrt(num_positions))
        spatial_sim = sim_matrix[label_idx, :]
        heatmap = spatial_sim.reshape(grid_size, grid_size)
        
        heatmap = heatmap.astype(np.float32)
        heatmap = cv2.resize(heatmap, tuple(original_img.shape[1:]), interpolation=cv2.INTER_LINEAR)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        cmap = plt.get_cmap('jet')
        heatmap_c = np.delete(cmap(heatmap), 3, 2)
        if len(original_img.shape) == 3 and original_img.shape[0] == 3:  # [C, H, W]
            original_img = np.transpose(original_img, (1, 2, 0))
        heatmap = 1*(1-heatmap**0.7)[..., np.newaxis]*original_img + \
                (heatmap**0.7)[..., np.newaxis] * heatmap_c
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # show overlay image
        ax.imshow(heatmap)
        ax.set_title(title)
        ax.axis('off')

    def denormalize(self, tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """denormalize a tensor image for visualization"""
        tensor = tensor.float().clone()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0, 1)

    def log_to_swanlab(self, fig, step, valid_texts, valid_semantics, args):
        """log the visualization figure and associated texts to SwanLab"""
        fig.canvas.draw()
        img_np = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        if args.writer:
            args.writer.log({"semantic_similarity": swanlab.Image(img_np, caption=f"Step {step}")})
            text_list = []
            for idx, t in enumerate(valid_texts):
                text_list.append(swanlab.Text(t, caption=f'{idx}'))
            if len(text_list) > 0:
                swanlab.log({"label": text_list})
            else:
                swanlab.log({"label": [swanlab.Text('None', caption='0')]})
            text_list = []
            for idx, t in enumerate(valid_semantics):
                text_list.append(swanlab.Text(t, caption=f'{idx}'))
            if len(text_list) > 0:
                swanlab.log({"semantic label": text_list})
            else:
                swanlab.log({"semantic label": [swanlab.Text('None', caption='0')]})