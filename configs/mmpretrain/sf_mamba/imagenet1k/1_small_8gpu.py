_base_ = [
    '../../_base_/datasets/imagenet_bs64_mambavision_224.py',
    '../../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../../_my_base_/default_runtime.py'
]

data_root = 'data/ImageNet/imagenet_orig/'

data_preprocessor = dict(
    type="DistilClsDataPreprocessor"
)


model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SFMamba',
        depths= [3, 3, 7, 5], dim=96, mid_patch_dim=64, drop_path_rate=0.2,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        in_channels=768,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        loss_distil=dict(type='MESADistilLoss', loss_weight=1., with_kl=True),
        num_classes=1000,
        topk=(
            1, 5,
        ),
        type='DistilLinearClsHead'),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=244,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCropV2',
        scale=224,
        backend='pillow',
        interpolation = ['bilinear', 'bicubic'],),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in _base_.bgr_mean], interpolation='bicubic')),
    # dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=_base_.bgr_mean,
        fill_std=_base_.bgr_std),
    dict(type='PackInputs'),
]

_base_.train_dataloader.batch_size = 128
_base_.train_dataloader.num_workers = 8
_base_.train_dataloader.dataset.data_root = data_root
_base_.train_dataloader.dataset.data_prefix = "train"
_base_.train_dataloader.dataset.ann_file='train/train.txt'
_base_.train_dataloader.dataset.pipeline=train_pipeline
_base_.val_dataloader.dataset.data_root = data_root
_base_.val_dataloader.dataset.data_prefix = "val"
_base_.val_dataloader.dataset.ann_file='val/val.txt'
_base_.val_dataloader.dataset.pipeline=test_pipeline
_base_.test_dataloader.dataset.data_root = data_root
_base_.test_dataloader.dataset.data_prefix = "val"
_base_.test_dataloader.dataset.ann_file='val/val.txt'
_base_.test_dataloader.dataset.pipeline=test_pipeline

optim_wrapper = dict(
    _delete_=True,
    type = "AmpOptimWrapper",
    loss_scale = 'dynamic',
    optimizer=dict(
        type='Lamb',
        lr=5e-3,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    accumulative_counts=4,
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
            '.dist_token': dict(decay_mult=0.0),
            '.A_log': dict(decay_mult=0.0),
            '.D.': dict(decay_mult=0.0),
            '.st_token': dict(decay_mult=0.0),
            '.ed_token': dict(decay_mult=0.0),
        }),
    clip_grad=dict(max_norm=5.0),
)

param_scheduler = [
    
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,
        convert_to_iter_based=True),
    
    dict(type='CosineAnnealingLR', eta_min=5e-6, by_epoch=True, begin=20, end=300)
]

train_cfg = dict(by_epoch=True, max_epochs=310, val_interval=1)



custom_hooks = [dict(type='MESA_EMAHook', momentum=1e-4, priority='ABOVE_NORMAL',
                     begin_distil_epoch=75, ema_type="mmdet.ExponentialMovingAverage",
                     evaluate_on_ema=True,
                     evaluate_on_origin=True,
                     )]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_best="auto",
    ),
)
