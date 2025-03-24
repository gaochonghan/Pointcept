"""
PTv3 + PPT
Pre-trained on ScanNet + Structured3D
(S3DIS is commented by default as a long data time issue of S3DIS: https://github.com/Pointcept/Pointcept/issues/103)
In the original PPT paper, 3 datasets are jointly trained and validated on the three datasets jointly with
one shared weight model. In PTv3, we trained on multi-dataset but only validated on one single dataset to
achieve extreme performance on one single dataset.

To enable joint training on three datasets, uncomment config for the S3DIS dataset and change the "loop" of
 Structured3D and ScanNet to 4 and 2 respectively.
"""

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 24  # bs: total bs in all gpus
num_worker = 48
mix_prob = 0.8
empty_cache = False
enable_amp = True
find_unused_parameters = True

# trainer
train = dict(
    type="MultiDatasetTrainer",
)

# model settings
model = dict(
    type="PPT-v1m1",
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=True,
        pdnorm_ln=True,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    backbone_out_channels=64,
    context_channels=256,
    conditions=("Structured3D", "ScanNet", "S3DIS"),
    template="[x]",
    clip_model="ViT-B/16",
    # fmt: off
    class_name=(
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
        "window", "bookshelf", "bookcase", "picture", "counter", "desk", "shelves", "curtain",
        "dresser", "pillow", "mirror", "ceiling", "refrigerator", "television", "shower curtain", "nightstand",
        "toilet", "sink", "lamp", "bathtub", "garbagebin", "board", "beam", "column",
        "clutter", "otherstructure", "otherfurniture", "otherprop",
    ),
    valid_index=(
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 33, 34, 35),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 20, 22, 24, 25, 27, 34),
        (0, 1, 4, 5, 6, 7, 8, 10, 19, 29, 30, 31, 32),
    ),
    # fmt: on
    backbone_mode=False,
)

# scheduler settings
epoch = 100
optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.005, 0.0005],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0005)]

# dataset settings
data = dict(
    num_classes=13,
    ignore_index=-1,
    names=[
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter",
    ],
    train=dict(
        type="ConcatDataset",
        datasets=[
            # Structured3D
            dict(
                type="Structured3DDataset",
                split=["train", "val", "test"],
                data_root="data/structured3d",
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2,
                    ),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(
                    #     type="ElasticDistortion",
                    #     distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                    # ),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                    # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", sample_rate=0.8, mode="random"),
                    dict(type="SphereCrop", point_max=204800, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    # dict(type="ShufflePoint"),
                    dict(type="Update", keys_dict={"condition": "Structured3D"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("color", "normal"),
                    ),
                ],
                test_mode=False,
                loop=4,  # sampling weight
            ),
            # ScanNet
            dict(
                type="ScanNetDataset",
                split="train",
                data_root="data/scannet",
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2,
                    ),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(
                    #     type="ElasticDistortion",
                    #     distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                    # ),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                    # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", point_max=102400, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    # dict(type="ShufflePoint"),
                    dict(type="Update", keys_dict={"condition": "ScanNet"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("color", "normal"),
                    ),
                ],
                test_mode=False,
                loop=2,  # sampling weight
            ),
            # S3DIS
            dict(
                type="S3DISDataset",
                split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
                data_root="data/s3dis",
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2,
                    ),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(
                    #     type="ElasticDistortion",
                    #     distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                    # ),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                    # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", sample_rate=0.6, mode="random"),
                    dict(type="SphereCrop", point_max=204800, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    # dict(type="ShufflePoint"),
                    dict(type="Update", keys_dict={"condition": "S3DIS"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("color", "normal"),
                    ),
                ],
                test_mode=False,
                loop=1,  # sampling weight
            ),
        ],
    ),
    val=dict(
        type="S3DISDataset",
        split="Area_5",
        data_root="data/s3dis",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={"coord": "origin_coord", "segment": "origin_segment"},
            ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"condition": "S3DIS"}),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "origin_coord",
                    "segment",
                    "origin_segment",
                    "condition",
                ),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type="S3DISDataset",
        split="Area_5",
        data_root="data/s3dis",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="Update", keys_dict={"condition": "S3DIS"}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "condition"),
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)
