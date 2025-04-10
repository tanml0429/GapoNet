import os
import sys
from pathlib import Path

# 获取文件夹下所有文件的数量
def count_dir(dir):
    dir = Path(dir)
    count = 0
    for file in dir.iterdir():
        if file.is_file():
            count += 1
    return count

if __name__ == '__main__':
    img_train_dir = '/data/tml/hybrid2_polyp_v5_format/images/results5'
    img_test_dir = '/data/tml/hybrid2_polyp_v5_format/images/results3'
    label_train_dir = '/data/tml/hybrid2_polyp_v5_format/labels/train'
    label_test_dir = '/data/tml/hybrid2_polyp_v5_format/labels/test'
    print('img_train_dir: {}'.format(count_dir(img_train_dir)))
    print('img_test_dir: {}'.format(count_dir(img_test_dir)))
    print('label_train_dir: {}'.format(count_dir(label_train_dir)))
    print('label_test_dir: {}'.format(count_dir(label_test_dir)))