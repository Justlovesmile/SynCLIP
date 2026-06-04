synclip_workdir="./logs"  # working directory for synclip training logs and checkpoints
data_root="./data/coco"   # root directory of coco dataset
pretrain_ckpt="./ckpts/EVA02_CLIP_L_336_psz14_s6B.pt"  # pretrained checkpoint for synclip training, can be downloaded from public link
vfm_type=dinov2-L         # {sam-B, sam-L, dinov2-B, dinov2-L, dino-B-8, dino-B-16}
exp_name="synclip_eva_l14_dinov2l_bs4x4_ep6_csa_csa_p_lsw10_k7_a7_b3_hk_C560"  # output folder name


# always keep total batchsize=16, otherwise, Linear scaling the learning rate (bs=4*4->lr=1e-5)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port 29500 -m training.main --batch-size=2 --lr=1e-5 --wd=0.1 --epochs=6 --workers=4 \
--model EVA02-CLIP-L-14-336 --pretrained eva --warmup 1000  --zeroshot-frequency 1 --dataset-type semantic_proposals_distill  \
--test-type coco_panoptic --train-data ${data_root}/Annotations/instances_train2017.json \
--val-data ${data_root}/Annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTL14x336.npy --train-image-root ${data_root}/Images/train2017 \
--val-image-root ${data_root}/Images/val2017 --cache-dir ${pretrain_ckpt} --log-every-n-steps 100 \
--lock-image --save-frequency 1 --lock-image-unlocked-groups 24 \
--name ${exp_name} --downsample-factor 14 --det-image-size 560 --val-segm-root  ${data_root}/Annotations/panoptic_val2017 \
--alpha 0.95  --use_vfm ${vfm_type} --mode csa_vfm_distill --loss_context_weight 0.05 --loss_content_weight 2.0 \
--swanlab --semantic-type semantic --max-semantic 20 --sem-mode csa --loss_semantic_weight 0.1 --use-topk --topk 7 \
--sem-ratio 0.7 --spa-ratio 0.3 --sem-loss huber_kl \
--semantic-path ${data_root}/Annotations/sevic.json \
--filelabel-path ${data_root}/Annotations/lvis_coco_categories_train2017.json
