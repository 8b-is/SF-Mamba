_base_ = [
    '../../../_base_/default_runtime.py', 
    '../../../_base_/datasets/ade20k.py',
    '../../../_base_/schedules/schedule_160k.py',
    '../../../_base_/models/upernet_swin.py',
]

data_root = "/data2/ADE20K/ADEChallengeData2016/"

checkpoint = 'work_dirs/mmpretrain/mamba_vision/imagenet1k/SF-Mamba_base.pth'


crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    init_cfg=dict(
        type="RevisePretrained", checkpoint=checkpoint, state_dict_key="ema_state_dict"),
    backbone=dict(
        _delete_=True,
        type='mmpretrain.SFMamba',
        depths= [3, 3, 10, 5], dim=128, mid_patch_dim=64, drop_path_rate=0.5,
        down_depthwise_conv=True,
        down_depthwise_ratio=2,
        window_size = (None, None, [64,64], [32,32]),
        classify=False,
        use_window=True,
    ),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=150),
    auxiliary_head=dict(in_channels=512, num_classes=150)
)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,                
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,                
    )
)
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,                
    )
)


optim_wrapper = dict(
    type='AmpOptimWrapper',
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
            '.norm': dict(decay_mult=0.0),
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.01))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=8000,
        max_keep_ckpts=1,
        save_best="auto",
    ),
)
