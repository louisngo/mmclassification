_base_ = [
    '../_base_/models/resnet_cifar_34_resnet_cifar_8_b128x1_st.py',
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
lr_config = dict(policy='step', step=[60, 120, 150], gamma=0.2)
runner = dict(type='EpochBasedRunner', max_epochs=200)