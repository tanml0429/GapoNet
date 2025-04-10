import os
import sys
from pathlib import Path


# 合并两个文件夹下的所有文件到一个文件夹
def merge_dir(dir1, dir2, out_dir):
    dir1 = Path(dir1)
    dir2 = Path(dir2)
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    for file in dir1.iterdir():
        if file.is_file():
            file_name = file.name
            file_path = file.absolute()
            os.system('cp {} {}'.format(file_path, out_dir))
    for file in dir2.iterdir():
        if file.is_file():
            file_name = file.name
            file_path = file.absolute()
            os.system('cp {} {}'.format(file_path, out_dir))



if __name__ == '__main__':
    img_train_dir1 = '/data/tml/splitdone_polyp_v5_format/dyed_endoscopy/images/train'
    img_train_dir2 = '/data/tml/splitdone_polyp_v5_format/normal_endoscopy/images/train'
    img_train_out_dir = '/data/tml/hybrid2_polyp_v5_format/images/train'
    merge_dir(img_train_dir1, img_train_dir2, img_train_out_dir)
    img_test_dir1 = '/data/tml/splitdone_polyp_v5_format/dyed_endoscopy/images/test'
    img_test_dir2 = '/data/tml/splitdone_polyp_v5_format/normal_endoscopy/images/test'
    img_test_out_dir = '/data/tml/hybrid2_polyp_v5_format/images/test'
    merge_dir(img_test_dir1, img_test_dir2, img_test_out_dir)
    label_train_dir1 = '/data/tml/splitdone_polyp_v5_format/dyed_endoscopy/labels/train'
    label_train_dir2 = '/data/tml/splitdone_polyp_v5_format/normal_endoscopy/labels/train'
    label_train_out_dir = '/data/tml/hybrid2_polyp_v5_format/labels/train'
    merge_dir(label_train_dir1, label_train_dir2, label_train_out_dir)
    label_test_dir1 = '/data/tml/splitdone_polyp_v5_format/dyed_endoscopy/labels/test'
    label_test_dir2 = '/data/tml/splitdone_polyp_v5_format/normal_endoscopy/labels/test'
    label_test_out_dir = '/data/tml/hybrid2_polyp_v5_format/labels/test'
    merge_dir(label_test_dir1, label_test_dir2, label_test_out_dir)



