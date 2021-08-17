_base_ = './logos_dataset_detection.py'

data = dict(
    train=dict(
        ann_file='ImageSets/Main/train.txt.bak.txt',
    ),
    val=dict(
        ann_file='ImageSets/Main/validation.txt.bak.txt',
    ),
    test=dict(
        ann_file='ImageSets/Main/test.txt.bak.txt',
    ),
)
