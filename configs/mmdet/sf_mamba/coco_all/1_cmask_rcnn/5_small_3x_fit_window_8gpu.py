_base_ = [
    '../../../_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '../../../_base_/datasets/coco_instance.py',
    '../../../_base_/schedules/schedule_3x.py', 
    '../../../_my_base_/default_runtime.py'
]

data_root = "data/coco/"

sync_bn='torch'
checkpoint = 'work_dirs/mmpretrain/mamba_vision/imagenet1k/SF-Mamba_small.pth'


model = dict(
    init_cfg=dict(
        type="RevisePretrained", checkpoint=checkpoint, state_dict_key="ema_state_dict"),

    backbone=dict(
        _delete_=True,
        type='mmpretrain.SFMamba',
        depths= [3, 3, 7, 5], dim=96, mid_patch_dim=64, drop_path_rate=0.2,
        window_size = (None, None, [84,84], [42,42]),
        classify=False,
        use_window=True,
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(bbox_head=[
        dict(
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
        dict(
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
        dict(
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.033, 0.033, 0.067, 0.067]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
    ]))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 1333), (500, 1333), (600, 1333)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]]),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        pipeline=train_pipeline,
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

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017.json',
)
test_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017.json',
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
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))