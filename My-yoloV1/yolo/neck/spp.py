import torch
import torch.nn as nn


class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x1, x2, x3], dim=1)
        return x
