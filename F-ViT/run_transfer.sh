bash dist_test.sh \
     configs/transfer/fvit_vitl14_upsample_fpn_transfer2objects365v1.py \
     work_dirs/fvit_synclip_eva_l14_dinov2l/epoch_48.pth  6  \
     --work-dir work_dirs/fvit_synclip_eva_l14_dinov2l/transfer \
     --eval bbox