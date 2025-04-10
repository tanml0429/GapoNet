import os
import sys
import argparse
import numpy as np
import pandas as pd
import shutil

from tqdm import tqdm


def make_dir(tp, trte):
    """创建文件夹"""
    if not os.path.exists(tp):
        os.makedirs(tp, exist_ok=True)  # 创建目录
    os.makedirs(f'{tp}/images/{trte}', exist_ok=True)
    os.makedirs(f'{tp}/labels/{trte}', exist_ok=True)

    # 清空文件夹
    for root, dirs, files in os.walk(f'{tp}/images/{trte}'):
        for file in files:
            os.remove(os.path.join(root, file))

    # 清空文件夹
    for root, dirs, files in os.walk(f'{tp}/labels/{trte}'):
        for file in files:
            os.remove(os.path.join(root, file))

def copy_files(img_path, tp, trte):
    # for filename in os.listdir(imgs_path):
            #img_path = os.path.join(imgs_path, filename)
        anno_path = img_path.replace('images', 'labels')
        if os.path.isfile(img_path) and os.path.isfile(anno_path):
            shutil.copy(img_path, f'{tp}/images/{trte}')
            shutil.copy(anno_path, f'{tp}/labels/{trte}')

# 随机取10%的数据作为测试集
def random_select(imgs_path, tp, trte, test_ratio=0.1):
    imgs_list = os.listdir(imgs_path)
    imgs_list.sort()
    import random
    random.shuffle(imgs_list)
    test_list = imgs_list[:int(len(imgs_list)*test_ratio)]
    return test_list

if __name__ == '__main__':
    tp = '/data/tml/dataset2'
    trte = 'train'
    make_dir(tp, trte)
    trte = 'test'
    make_dir(tp, trte)
    
    for trte in ['train']:
        imgs_path_list = ['/data/tml/mixed_polyp_v5_format/PolypsSet/images', '/data/tml/mixed_polyp_v5_format/LDTrainValid/images']
        for imgs_path in imgs_path_list:
            for filename in os.listdir(imgs_path):
                img_path = os.path.join(imgs_path, filename)
                copy_files(img_path, tp, trte)
        

    for trte in ['test']:
        imgs_path = '/data/tml/mixed_polyp_v5_format/LDTrainValid/images'
        test_list = random_select(imgs_path, tp, trte, test_ratio=0.1)
        for filename in test_list:
            img_path = os.path.join(imgs_path, filename)
            copy_files(img_path, tp, trte)

        imgs_path_list = ['/data/tml/mixed_polyp_v5_format/BKAI-IGH_NeoPolyp-Small/images', '/data/tml/mixed_polyp_v5_format/Dataset-acess-for-PLOS-ONE/images']
        for imgs_path in imgs_path_list:
            for filename in os.listdir(imgs_path):
                img_path = os.path.join(imgs_path, filename)
                copy_files(img_path, tp, trte)