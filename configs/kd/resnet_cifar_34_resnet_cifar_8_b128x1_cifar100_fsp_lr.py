_base_ = [
    '../_base_/datasets/cifar100_bs16.py'
]

# checkpoint saving
checkpoint_config = dict(interval=50)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# optimizer
optimizer = dict(type='SGD',
                 lr=0.1, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[120, 150], gamma=0.1,
                 warmup='constant', warmup_iters=100, warmup_ratio=0.01,
                 warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=200)


# model settings
model = dict(
    type='BaseClassifierKD',
    kd_loss=dict(type='FspLoss'),
    train_cfg=dict(lambda_kd=1.0,
                   kd_loss_2=dict(type='KnowledgeDistillationKLDivLoss', T=4.0),
                   end_of_first_stage=100,
                   teacher_checkpoint="work_dirs/resnet_cifar_34_b128x1_cifar100/epoch_200.pth"), # Input your teacher checkpoint
    backbone=dict(
        # return_tuple=False,
        student=dict(
            type='ResNet_CIFAR',
            depth=8,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            style='pytorch'),
        teacher=dict(
            type='ResNet_CIFAR',
            depth=34,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            style='pytorch')
    ),
    neck=dict(
        student=dict(type='GlobalAveragePooling'),
        teacher=dict(type='GlobalAveragePooling')
    ),
    head=dict(
        student=dict(
            type='LinearClsHead',
            num_classes=100,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        ),
        teacher=dict(
            type='LinearClsHead',
            num_classes=100,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        )
    ))
