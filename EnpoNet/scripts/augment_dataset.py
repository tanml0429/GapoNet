"""
对labelme标注的数据集进行扩增
方法：动态背景、随机尺寸、随机角度、随机位置
"""

import os, sys
from pathlib import Path
pydir = Path(os.path.abspath(__file__)).parent

try:
    import damei as dm
except:
    sys.path.append(f'{pydir.parent.parent}/damei')
    import damei as dm
import argparse


def augment(opt):
    # 函数参数含义可查阅damei lib文档：http://47.114.37.111
    dm.data.augment(
        source_path=opt.sp,  # labelme标注的数据集路径
        target_path=opt.tp,  # 输出的数据集路径，默认为source_path同级目录下的xxx_augmented_YOLOfmt文件夹
        backgrounds_path=opt.bp,  # 额外的背景图片路径，默认无
        anno_fmt=opt.anno_fmt,  # 标注文件格式
        out_fmt=opt.out_fmt,  # 输出文件格式
        use_noise_background=opt.use_noise_background,  # 是否使用噪声背景，默认不使用
        out_size=opt.out_size,  # 输出图片尺寸
        num_augment_images=opt.num_augment_images,  # 扩增图片数量
        train_ratio=opt.train_ratio,  # 训练集比例，其余为测试集
        erase_ratio=opt.erase_ratio,  # 随机删除原始标注中的目标的比例范围，(0.5, 1.0)表示对每张图随机删除，50%~100%的目标，平均随机
        mean_scale_factor=opt.mean_scale_factor,  # 图片缩放的均值
        save_mask=opt.save_mask,  # 合成中保存中间每个目标的蒙版（会占用大量硬盘空间）
        iou_threshold=opt.iou_threshold,  # 合成过程中新增的目标与已存在目标的IOU阈值，小于该阈值会合成，大于该阈值会重新随机，随机10次依然不满足阈值则不合成
        suffix=opt.suffix,  # 原始图像后缀
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Augment')
    parser.add_argument('--sp', default=None, help='labelme标注的数据集路径')
    parser.add_argument('--tp', default=None, help='输出的数据集路径，默认为source_path同级目录下的xxx_augmented_YOLOfmt文件夹')
    parser.add_argument('--bp', default=None, help='额外的背景图片路径，默认None，即无额外的背景')
    parser.add_argument('--anno-fmt', default='labelme', help='标注文件格式')
    parser.add_argument('--out-fmt', default='YOLOfmt', help='输出文件格式')
    parser.add_argument('--use-noise-background', action='store_true', help='是否使用噪声背景，默认不使用')
    parser.add_argument('--out-size', type=int, default=640, help='输出图片尺寸')
    parser.add_argument('--num-augment-images', type=int, default=5000, help='扩增图片总数量')
    parser.add_argument('--train-ratio', type=float, default=0.8, help="训练集比例，其余为测试集")
    parser.add_argument('--erase-ratio', default=0., help="随机删除原始标注中的目标的比例范围，0表示删除，(0.5, 1.0)表示对每张图随机删除，50%~100%的目标，平均随机")
    parser.add_argument('--mean-scale-factor', default=0.3, help='图片缩放的均值')
    parser.add_argument('--iou-threshold', default=0.3, help='合成过程中新增的目标与已存在目标的IOU阈值，小于该阈值会合成，大于该阈值会重新随机，随机10次依然不满足阈值则不合成')
    parser.add_argument('--save-mask', action='store_true', help='合成中保存中间每个目标的蒙版（会占用大量硬盘空间）')
    parser.add_argument('--suffix', type=str, default='.jpg', help='图像后缀')
    opt = parser.parse_args()

    opt.out_size = 1280
    opt.num_augment_images = 8000
    opt.erase_ratio = (0.5, 1.0)
    opt.sp = f'{Path.home()}/datasets/longhu/datasets/jinglin/raw_80'
    opt.tp = f'{Path.home()}/datasets/longhu/datasets/jinglin/augment_YOLOfmt_{opt.num_augment_images}'

    augment(opt)





