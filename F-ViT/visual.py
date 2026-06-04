import os
import json
import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

def visualize_coco_annotation(
    img_name,
    img_dir,
    ann_file,
    show_bbox=True,
    show_mask=True,
    save_path=None,
    color_dict=None
):
    """
    可视化COCO标注结果
    Args:
        img_name (str): 图像文件名，例如 "000000397133.jpg"
        img_dir (str): 图像所在目录
        ann_file (str): COCO 标注文件路径 (JSON)
        show_bbox (bool): 是否显示bbox
        show_mask (bool): 是否显示分割mask
        save_path (str): 若不为None，则保存绘制结果
    """
    # 初始化 COCO api
    if type(ann_file) == str:
        coco = COCO(ann_file)
    else:
        coco = ann_file
    # 根据文件名找到对应图像id
    if 'file_name' in coco.dataset['images'][0].keys():
        img_info = next((img for img in coco.dataset["images"] if img["file_name"] == img_name), None)
    else:
        img_info = next((img for img in coco.dataset["images"] if os.path.basename(img["coco_url"]) == img_name), None)
        if "train2017" in img_info["coco_url"]:
            img_name = "train2017/"+img_name
        elif "val2017" in img_info["coco_url"]:
            img_name = "val2017/"+img_name
    if img_info is None:
        print(f"未找到图像: {img_name}")
        return
    
    img_id = img_info["id"]
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)

    # 读取图像
    img_path = os.path.join(img_dir, img_name)
    img = np.array(Image.open(img_path).convert("RGB"))

    # 绘图
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    for ann in anns:
        cat = coco.loadCats(ann["category_id"])[0]["name"]
        if color_dict is None:
            color = np.random.rand(3)
        else:
            color = color_dict[cat]
        
        # 绘制分割mask
        if show_mask:
            if isinstance(ann["segmentation"], list):
                for seg in ann["segmentation"]:
                    poly = np.array(seg).reshape((len(seg)//2, 2))
                    patch = patches.Polygon(poly, facecolor=color, edgecolor=color, alpha=0.4)
                    ax.add_patch(patch)
            elif isinstance(ann["segmentation"], dict):
                rle = maskUtils.frPyObjects(ann["segmentation"], img.shape[0], img.shape[1])
                m = maskUtils.decode(rle)
                ax.imshow(m, alpha=0.4, cmap='jet')
        
        # 绘制bbox
        if show_bbox:
            x, y, w, h = ann["bbox"]
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y - 2, cat, color='white', fontsize=10,
                    bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1))
    
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        # print(f"结果已保存到: {save_path}")
    else:
        plt.show()
        
        
if __name__=="__main__":
    coco = COCO("/mnt/DataShare/_Public_Datasets_/Detection/COCO2017/zero-shot/instances_val2017_all_2.json")
    seen_classes='/home/sgiit/SGIIT/xmj/SynCLIP/F-ViT/datasets/mscoco_seen_classes.json'
    all_classes='/home/sgiit/SGIIT/xmj/SynCLIP/F-ViT/datasets/mscoco_all_classes.json'
    save_dir = '/home/sgiit/SGIIT/xmj/SynCLIP/F-ViT/work_dirs/fvit_synclip_eva_l14_dinov2l_bs6x2_ep6_csa_csa_p_lsw10_k7_a7_b3_hk_C560_ds_F896_COCO/ground_truth'
    os.makedirs(save_dir, exist_ok=True)
    seen_classes = json.load(open(seen_classes)) + ['background']
    all_classes = json.load(open(all_classes)) + ['background']
    unseen_classes = [c for c in all_classes if c not in seen_classes]
    print("seen_classes:", seen_classes)
    print("unseen_classes:", unseen_classes)
    color_dict = {}
    for c in all_classes:
        if c in seen_classes:
            color_dict[c] = (0, 0, 1)
        else:
            color_dict[c] = (1, 1, 0)
    for img in tqdm.tqdm(coco.dataset['images']):
        if 'file_name' in img.keys():
            filename = img['file_name']
        else:
            filename = os.path.basename(img['coco_url'])
        visualize_coco_annotation(
            img_name=filename,
            img_dir="/mnt/DataShare/_Public_Datasets_/Detection/COCO2017/Images/val2017/",
            ann_file=coco,
            show_bbox=True,
            show_mask=True,
            save_path=os.path.join(save_dir, filename),
            color_dict=color_dict
        )