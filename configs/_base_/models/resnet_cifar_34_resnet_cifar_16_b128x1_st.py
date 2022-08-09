# model settings
model = dict(
    type='BaseClassifierKD',
    kd_loss=dict(type='KnowledgeDistillationKLDivLoss', T=4.0),
    train_cfg=dict(lambda_kd=0.9,
                   teacher_checkpoint="work_dirs/resnet_cifar_34_b128x1_cifar100/epoch_200.pth"), # Input your teacher checkpoint
    backbone=dict(
        # return_tuple=False,
        student=dict(
            type='ResNet_CIFAR',
            depth=16,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        teacher=dict(
            type='ResNet_CIFAR',
            depth=34,
            num_stages=4,
            out_indices=(3, ),
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
