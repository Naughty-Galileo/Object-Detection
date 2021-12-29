from __future__ import division

import os
import random
import argparse
import time
import math
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision

from data import *
from yolo import *
from utils import *
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('-v', '--version', default='yolo',
                        help='yolo')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=1,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                        default=1, help='interval between evaluations')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=True,
                        help='use tensorboard')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights/', type=str,
                        help='Gamma update for SGD')

    return parser.parse_args()


def train():
    # 创建命令行参数
    args = parse_args()
    print("Setting Arguments...:", args)
    print("-----------------------------------")

    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # 是否启用cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # 是否使用多尺度训练
    if args.multi_scale:
        print("use the multi-scale trick...")
        train_size = 640
        val_size = 416
    else:
        train_size = 416
        val_size = 416

    # 训练用到的配置参数
    cfg = train_cfg

    # 创建dataset类和dataloader类
    if args.dataset == 'voc':
        # 加载VOC数据集
        data_dir = VOC_ROOT
        num_classes = 20
        dataset = VOCDetection(root=data_dir, transform=SSDAugmentation(train_size))
        evaluator = VOCAPIEvaluator(data_root=data_dir,
                                    img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size),
                                    labelmap=VOC_CLASSES)

    print('Train model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("------------------------------------------")

    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, collate_fn=detection_collate,
                            num_workers=args.num_workers, pin_memory=True)

    # 创建模型
    yolo_net = MyYolo(device=device,
                      input_size=train_size,
                      num_classes=num_classes,
                      trainable=True)
    model = yolo_net
    model.to(device).train()

    # 是否使用tensorboard来保存训练过程中的各类数据
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        # c_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
        # log_path = os.path.join('./log/', c_time)
        # os.makedirs(log_path='./log', exist_ok=True)

        writer = SummaryWriter('./log')

    if args.resume is not None:
        print('keep training model: %s' % args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # 构建训练所用的优化器
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    max_epoch = cfg['max_epoch']  # 最大训练次数
    epoch_size = len(dataset) // args.batch_size  # 每一训练轮次的迭代次数

    # 开始训练
    for epoch in range(args.start_epoch, max_epoch):

        # 阶梯式学习衰减策略
        if epoch in cfg['lr_epoch']:
            tmp_lr *= 0.1
            set_lr(optimizer, tmp_lr)

        # 获取一批数据
        for iter_i, (images, targets) in enumerate(dataloader):
            # 使用warm-up学习率策略
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    ni = iter_i + epoch * epoch_size
                    nw = args.wp_epoch * epoch_size
                    tmp_lr = base_lr * pow(ni / nw, 4)
                    set_lr(optimizer, tmp_lr)
                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)

            # 多尺度训练
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # 随机选取新的训练尺度
                train_size = random.randint(10, 19) * 32
                model.set_grid(train_size)
            if args.multi_scale:
                images = torch.nn.functional.interpolate(images,
                                                         size=train_size,
                                                         mode='bilinear',
                                                         align_corners=False
                                                         )
            # 制作训练标签
            targets = [label.tolist() for label in targets]
            targets = gt_creator(input_size=train_size,
                                 stride=yolo_net.stride,
                                 label_lists=targets)

            # to device
            images = images.to(device)
            targets = targets.to(device)

            # 前向推理，计算损失
            conf_loss, cls_loss, bbox_loss, total_loss = model(images, targets)
            if iter_i % 10 == 0:
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                      '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d]'
                      % (epoch + 1, max_epoch, iter_i, epoch_size, tmp_lr,
                         conf_loss.item(),
                         cls_loss.item(),
                         bbox_loss.item(),
                         total_loss.item(),
                         train_size),
                      flush=True)

                if args.tfboard:
                    # viz loss
                    writer.add_scalar('obj loss', conf_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('cls loss', cls_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('box loss', bbox_loss.item(), iter_i + epoch * epoch_size)

            # 反向传播和更新
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 验证模型性能
        model.trainable = False
        model.set_grid(val_size)
        model.eval()

        evaluator.evaluate(model)

        # 转变为训练模式
        model.trainable = True
        model.set_grid(train_size)
        model.train()

        # 保存模型
        print('Saving state, epoch:', epoch + 1)
        torch.save(model.state_dict(), os.path.join(path_to_save, args.version + '_'
                                                    + repr(epoch + 1) + '.pth'))


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
