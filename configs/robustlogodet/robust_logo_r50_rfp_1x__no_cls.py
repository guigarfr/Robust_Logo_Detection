_base_ = [
    '../_base_/models/robust_logo_r50_rfp__no_cls.py',
    '../_base_/datasets/logos_dataset_detection_nc.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# Modify grad clip
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=5, norm_type=2))

# Change learning policy step
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 12, 16])

# Set max epochs to 18
runner = dict(type='EpochBasedRunner', max_epochs=18)

work_dir = "/home/ubuntu/logos_dataset/checkpoints_rp_no_class"
load_from = "/home/ubuntu/epoch_21.pth"
data_root = '/home/ubuntu/data/logo_dataset/'
