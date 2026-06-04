_base_ = 'fvit_vitb16_upsample_fpn_bs64_4x_ovlvis_eva_original.py'
model = dict(
    backbone=dict(
        pretrained='checkpoints/DeCLIP_EVA-B_DINOv2-B_csa_0.05_2.0.pt'),
)
