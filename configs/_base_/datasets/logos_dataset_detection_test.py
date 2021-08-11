_base_ = [
    '../_base_/datasets/logos_dataset_detection.py'
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Main/train.txt.bak.txt',
        img_prefix='',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Main/validation.txt.bak.txt',
        img_prefix='',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Main/test.txt.bak.txt',
        img_prefix='',
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric='bbox')
