synclip_workdir="/home/xmj/SynCLIP/logs"
exp_name="synclip_eva_b16_dinov2l_bs4x4_ep6_csa_csa_p_lsw10_k7_a7_b3_hk_C560"
CKPT=${synclip_workdir}/${exp_name}/checkpoints/epoch_6.pt

if [ -f "$CKPT" ]; then
    echo "[INFO] 找到 checkpoint: $CKPT"

    bash dist_train.sh configs/ov_coco/fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_synclip_proposals.py 4 --auto-resume \
    --cfg-options model.backbone.pretrained=${synclip_workdir}/${exp_name}/checkpoints/epoch_6.pt model.roi_head.bbox_head.vlm_temperature=75.0 model.roi_head.bbox_head.beta=0.8 \
    --work-dir ./work_dirs/fvit_${exp_name}_F640_COCO

else
    echo "[WARN] $CKPT 不存在"
fi