import numpy as np
import torch
import torch.nn as nn
from .backbone.resnet import resnet18
from .neck.spp import SPP
from utils.tools import loss


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


class MyYolo(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False,
                 conf_thresh=0.01, nms_thresh=0.5):
        super(MyYolo, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = 32  # 最大的下采样倍数  最大步长
        self.grid_cell = self.create_grid(input_size)  # 得到最终的bbox的参数 网格坐标矩阵
        self.input_size = input_size

        #                 backbone                #
        self.backbone = resnet18(pretrained=True)
        c5 = 512
        #                 neck                    #
        self.neck = nn.Sequential(SPP(),
                                  Conv(c5 * 4, c5, k=1))
        #                 head                    #
        self.convsets = nn.Sequential(
            Conv(c5, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, p=1)
        )
        #                 pred                    #
        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)

    def set_grid(self, input_size):
        """
            用于重置G矩阵。
        """
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)

    def create_grid(self, input_size):
        # 输入图像的宽和高
        w, h = input_size, input_size
        # 特征图的宽和高
        ws, hs = w // self.stride, h // self.stride
        # 使用torch.meshgrid获取矩阵G的x坐标和y坐标
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        # 将xy拼接在一起，得到矩阵G
        # 13*13*2 (输入416/下采样32=13)
        #  [[[0,0],[1,0],[2,0],…,[12,0]], [[0,1],[1,1],[2,1],…,[12,1]], … , [[0,12],[1,12],[2,12],…,[12,12]]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        # 最终G的维度[1,HW,2]
        # 1*169*2
        grid_xy = grid_xy.view(1, hs * ws, 2).to(self.device)

        return grid_xy

    def decode_boxes(self, pred):
        # 将网络输出的tx,ty,tw,th四个量转换成bbox的(x1,y1),(x2,y2)
        output = torch.zeros_like(pred)
        # 获取bbox的中心点坐标和宽高
        pred[:, :, :2] = (torch.sigmoid(pred[:, :, :2]) + self.grid_cell) * self.stride
        pred[:, :, 2:] = torch.exp(pred[:, :, 2:])
        # 由中心点坐标和宽高获得左上角与右下角的坐标
        output[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
        output[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
        output[:, :, 2] = pred[:, :, 0] + pred[:, :, 2] / 2
        output[:, :, 3] = pred[:, :, 1] + pred[:, :, 3] / 2

        return output

    def nms(self, dets, scores):
        x1 = dets[:, 0]  # xmin
        y1 = dets[:, 1]  # ymin
        x2 = dets[:, 2]  # xmax
        y2 = dets[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)  # 计算每个bbox的面积
        order = scores.argsort()[::-1]  # 按照降序对bbox的得分进行排序 序号排序

        keep = []
        while order.size > 0:
            i = order[0]  # 得到最高的那个bbox
            keep.append(i)  # 最大的保留下来

            xx1 = np.maximum(x1[i], x1[order[1:]])  # 在两个多维数组中逐元素求最大值和最小值
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h  # 交集

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, bboxes, scores):
        cls_inds = np.argmax(scores, axis=1)

        scores = scores[(np.arange(scores.shape[0]), cls_inds)]

        # 进行阈值筛选，滤除得分低的检测框
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # 对每一类进行NMS操作
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        # 获得最终的检测结果
        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds

    def forward(self, x, target=None):
        c5 = self.backbone(x)

        p5 = self.neck(c5)

        p5 = self.convsets(p5)

        pred = self.pred(p5)
        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        pred = pred.view(p5.size(0), 1 + self.num_classes + 4, -1).permute(0, 2, 1)

        # 从预测的pred中分理处object、class、txtytwth三部分的预测
        # object预测：[B, H*W, 1]
        conf_pred = pred[:, :, :1]
        # class预测：[B, H*W, num_cls]
        cls_pred = pred[:, :, 1: 1 + self.num_classes]
        # bbox预测：[B, H*W, 4]
        txtytwth_pred = pred[:, :, 1 + self.num_classes:]

        if self.trainable:
            conf_loss, cls_loss, bbox_loss, total_loss = loss(pred_conf=conf_pred,
                                                              pred_cls=cls_pred,
                                                              pred_txtytwth=txtytwth_pred,
                                                              label=target)
            return conf_loss, cls_loss, bbox_loss, total_loss

        else:
            with torch.no_grad():
                # 去掉batch维度
                conf_pred = torch.sigmoid(conf_pred)[0]
                bboxes = torch.clamp((self.decode_boxes(txtytwth_pred) / self.input_size)[0], 0., 1.)
                scores = (torch.softmax(cls_pred[0, :, :], dim=1) * conf_pred)

                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()
                # 后处理
                bboxes, scores, cls_inds = self.postprocess(bboxes, scores)
                return bboxes, scores, cls_inds
