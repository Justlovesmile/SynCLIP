workdir="./work_dirs/fvit_synclip_eva_l14_dinov2l"

bash dist_test.sh \
    $workdir/fvit_vitl14_upsample_fpn_bs64_4x_ovlvis_eva_synclip.py \
    $workdir/epoch_48.pth 6 \
    --work-dir $workdir/eval \
    --out $workdir/eval/results.pkl