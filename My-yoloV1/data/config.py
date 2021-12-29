# config.py
import os.path

# new yolo config
train_cfg = {
    'lr_epoch': (3, 8),
    'max_epoch': 10,
    'min_dim': [416, 416]
}