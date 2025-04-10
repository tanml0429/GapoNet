
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File      :   HWDown.py
@Time      :   2024/04/04 23:22:13
@Author    :   CSDN迪菲赫尔曼 
@Version   :   1.0
@Reference :   https://blog.csdn.net/weixin_43694096
@Desc      :   None
"""


import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward  # pip install pytorch_wavelets


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode="zero", wave="haar")
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        device = x.device  # 动态获取输入张量的设备
        self.wt.to(device)  # 将变换层移至正确的设备
        self.conv_bn_relu.to(device)  # 将卷积层移至正确的设备

        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, :]
        y_LH = yH[0][:, :, 1, :]
        y_HH = yH[0][:, :, 2, :]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

