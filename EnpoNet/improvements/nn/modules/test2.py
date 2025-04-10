import torch
import torch.nn as nn
from mamba import SS2D

if __name__ == '__main__':
    ss2d = SS2D(d_model=12).cuda()
    x = torch.randn(1, 12, 640, 640) # batch_size, channels, height, width
    x = x.cuda()
    y = ss2d(x)
    print(y.shape)

