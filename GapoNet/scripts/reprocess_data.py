import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageEnhance
import colorsys

class Reprocess(object):
    def __init__(self):
        pass
    def __call__(self):
        pass

    def rm_highlight(self, img_path):
        img = cv2.imread(img_path)
        img_name = img_path.split('/')[-1]
        # 将图像转为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 对灰度图像进行直方图均衡化
        equ = cv2.equalizeHist(gray)

        # 通过高斯滤波平滑图像
        blur = cv2.GaussianBlur(equ, (5,5), 0)

        # 检测边缘
        canny = cv2.Canny(blur, 30, 150)

        # 显示结果
        # cv2.imshow('Original Image', img)
        # cv2.imshow('Processed Image', canny)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # 自适应阈值二值化
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # 膨胀和腐蚀操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilate = cv2.dilate(canny, kernel, iterations=1)
        erode = cv2.erode(dilate, kernel, iterations=1)

        # 寻找轮廓
        contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 面积筛选
        # for i in range(len(contours)):
            # area = cv2.contourArea(contours[i])  # 计算轮廓面积
            # if area < 100:
                # cv2.drawContours(erode, [contours[i]], 0, 0, -1)  # 将面积小的轮廓涂黑

        # 显示结果
        # cv2.imshow('Original Image', img)
        # cv2.imshow('Processed Image', erode)

        return img, img_name, gray, contours

    def fill_highlight(self, img, img_name, gray, contours):
        # 找到最大轮廓
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # 填充高光区域
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [max_contour], 0, 255, -1)
        result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        # 显示结果
        # cv2.imshow('Original Image', img)
        # cv2.imshow('Result Image', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 根据图片名保存结果
        cv2.imwrite(f'{Path(img_path).parent.parent}/results1/{img_name}', result)

        
    
    def rm_shadow(self, img_path, img_name):
        img = cv2.imread(img_path)
        # 将图片转换为HSV颜色空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 提取彩色光伪影的掩码
        lower_range = np.array([100, 100, 150])  # 设置下限阈值
        upper_range = np.array([180, 255, 255])  # 设置上限阈值
        mask = cv2.inRange(hsv, lower_range, upper_range)

        # 使用掩码去除彩色光伪影
        result = cv2.bitwise_and(img, img, mask=mask)


        # 对图像进行填充
        border_size = 50  # 边缘填充大小
        border_color = [255, 255, 255]  # 填充颜色
        result = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)


        # 显示原始图片和处理后的图片
        # cv2.imshow('Original Image', img)
        # cv2.imshow('Processed Image', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 保存结果
        cv2.imwrite(f'{Path(img_path).parent.parent}/results2/{img_name}', result)

    def enhance(self, img_path, img_name):
        # 读入图片
        img = cv2.imread(img_path)

        # 将图像转换为 8 位无符号整数深度
        img = cv2.convertScaleAbs(img, cv2.IMREAD_UNCHANGED)

        # 去噪声
        # img = cv2.medianBlur(img, 3)

        # 归一化像素值
        # img = img / 255.0

        # 计算图像的平均亮度和标准差
        mean = np.mean(img)
        std = np.std(img)

        # 扩大亮度和对比度的范围
        # alpha = 0.1 * (1 / std)
        # beta = (1 * (0.5 - alpha * mean)) / (1 - alpha)
        alpha = 1.2
        beta = 0

        # 调整对比度和亮度
        img = np.clip(alpha * img + beta, 0, 255)

        # 锐化图像
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # img = cv2.filter2D(img, -1, kernel)

        # 还原像素值范围
        # img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

        # 增强颜色饱和度,不超过255
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:,:,1] = np.clip(img_hsv[:,:,1]*1.1, 0, 255)
        img_hsv[:,:,2] = np.clip(img_hsv[:,:,2]*1.5, 0, 255)

        # 还原BGR颜色空间
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

       

        

    

        # 保存结果
        cv2.imwrite(f'{Path(img_path).parent.parent}/results3/{img_name}', img)

    def enhance2(self, img_path, img_name):
        img = Image.open(img_path)
        enh_bri = ImageEnhance.Brightness(img)
        brightness = 1.1
        img = enh_bri.enhance(brightness)
        enh_con = ImageEnhance.Contrast(img)
        contrast = 1.5
        img = enh_con.enhance(contrast)
        enh_col = ImageEnhance.Color(img)
        color = 0.8
        img = enh_col.enhance(color)
        enh_sha = ImageEnhance.Sharpness(img)
        sharpness = 2
        img = enh_sha.enhance(sharpness)

        

        # 将图像转换为 HSV 颜色空间
        img_hsv = img.convert('HSV')

        # 调整饱和度
        hue, saturation, value = colorsys.rgb_to_hsv(*(map(lambda i: i / 255.0, img.getpixel((1, 1)))))  # 获取原图像素值
        img_hsv = img_hsv.convert('HSV')
        img_hsv = img_hsv.point(lambda i: i * 1.5 if i == saturation else i)  # 增强饱和度
        img_rgb = img_hsv.convert('RGB')

        # 保存结果
        img_rgb.save(f'{Path(img_path).parent.parent}/results5/{img_name}')

    def rm_highlight2(self, img_path, img_name):
        # 读取图像
        img = cv2.imread(img_path)

        # 计算阈值
        threshold = 0.8 * np.max(img)

        # 提取反光像素
        mask = img > threshold

        # 中值滤波
        kernel_size = 21
        img_median = cv2.medianBlur(img, kernel_size)

         # 替换反光像素
        img_filtered = np.where(mask, img_median, img)

        # 腐蚀
        kernel_size = 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        img_erode = cv2.erode(img_filtered, kernel, iterations=1)

        # 膨胀
        kernel_size = 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        img_dilate = cv2.dilate(img_erode, kernel, iterations=3)

        # 形态学闭运算
        closing_kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
        img_closed = cv2.morphologyEx(img_dilate, cv2.MORPH_CLOSE, kernel)

        
        

        # 保存结果
        cv2.imwrite(f'{Path(img_path).parent.parent}/results6/{img_name}', img_closed)
        img_path1 = f'{Path(img_path).parent.parent}/results6/{img_name}'

        return img_path1

    
if __name__ == '__main__':
    imgs_path = '/data/tml/test_mask/BKAI-IGH_NeoPolyp-Small/images'
    for i in os.listdir(imgs_path):
        img_path = os.path.join(imgs_path, i)
        img_name = img_path.split('/')[-1]
        reprocess = Reprocess()
        # img, img_name, gray, contours = reprocess.rm_highlight(img_path)
        # result = reprocess.fill_highlight(img, img_name, gray, contours)
        # result = reprocess.rm_shadow(img_path, img_name)
        
        img_path1 = reprocess.rm_highlight2(img_path, img_name)
        reprocess.enhance2(img_path1, img_name)
    
    



