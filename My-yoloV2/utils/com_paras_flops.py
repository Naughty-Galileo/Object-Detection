import torch
from thop import profile
from yolo.yolov2 import YOLOv2 as net
from data import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    input_image = torch.randn(1, 3, 640, 640).to(device)
    input_size = 640
    num_classes = 20
    model = net(device, input_size=input_size, num_classes=num_classes, anchor_size=ANCHOR_SIZE).to(device)
    flops, params = profile(model, inputs=(input_image,))
    print('FLOPs : ', flops / 1e9, ' B')
    print('Params : ', params / 1e6, ' M')


if __name__ == "__main__":
    main()
