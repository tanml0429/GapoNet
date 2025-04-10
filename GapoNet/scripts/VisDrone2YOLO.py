"""
VisDrone数据集转换为YOLO格式的数据集。
"""
import os, sys
from pathlib import Path
import shutil
import argparse
import cv2
import damei as dm
import numpy as np
logger = dm.getLogger('VisDrone2YOLO')


class VisDrone2YOLO(object):

    def __init__(self, opt):
        """
        # VisDrone的标注格式：xmin, ymin, width, height, score, category, truncation, occlusion
        # score: 0-1, 0是评估中不考虑的，1是考虑的
        # category: 0-11: ignored regions (0), pedestrian (1), people (2), bicycle (3), car (4), van (5),
        #                 truck (6), tricycle (7), awning-tricycle (8), bus (9), motor (10), others (11))
        # truncation: 0-1, no truncation = 0 (truncation ratio 0%),
        #                 and partial truncation = 1(truncation ratio 1% ∼ 50%)
        # occlusion: 0-1, no occlusion = 0 (occlusion ratio 0%),
        #                 partial occlusion = 1(occlusion ratio 1% ∼ 50%)
        #                 heavy occlusion = 2 (occlusion ratio 50% ~ 100%)
        """
        self.opt = opt
        self.classes = ["ignored regions", "pedestrian", "people", "bicycle", "car", "van",
                        "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"]

    def visdrone2yolo(self):
        """
        将VisDrone数据集转换为YOLO格式的数据集。
        :param visdrone_path: VisDrone数据集路径。
        :param yolo_path: YOLO格式的数据集路径。
        :return: None
        """
        opt = self.opt
        sp = opt.visdrone_path
        tp = opt.yolo_path

        if os.path.exists(tp):
            if not opt.force:
                raise FileExistsError('YOLO格式的数据集已存在，请先删除或-f强制覆盖。')
            shutil.rmtree(tp)
        os.makedirs(tp)
        os.makedirs(f'{tp}/images/train')
        os.makedirs(f'{tp}/images/test')
        os.makedirs(f'{tp}/labels/train')
        os.makedirs(f'{tp}/labels/test')
        with open(f'{tp}/classes.txt', 'w') as f:
            f.writelines('\n'.join(self.classes))

        # 将VisDrone数据集转换为YOLO格式的数据集。
        trtes = ['train', 'test']
        for i, trte in enumerate(trtes):
            sptr = 'VisDrone2019-DET-train' if trte == 'train' else 'VisDrone2019-DET-val'
            imgs = sorted([x for x in os.listdir(f'{sp}/{sptr}/images') if x.endswith('.jpg')])
            labels = sorted([x for x in os.listdir(f'{sp}/{sptr}/annotations') if x.endswith('.txt')])
            print(f'{len(imgs)} images and {len(labels)} labels in {sptr}.')
            for j, (img_name, label) in enumerate(zip(imgs, labels)):
                assert img_name == label.replace('.txt', '.jpg'), f'stem: {img_name} != {label}'
                assert os.path.exists(f'{sp}/{sptr}/images/{img_name}'), f'{sp}/{sptr}/images/{img_name} not exists.'

                img = cv2.imread(f'{sp}/{sptr}/images/{img_name}')
                h, w = img.shape[:2]
                with open(f'{sp}/{sptr}/annotations/{label}', 'r') as f:
                    lines = f.readlines()
                lines = [x.strip() for x in lines]
                new_lines = []
                for line in lines:
                    line = line[:-1].split(',') if line.endswith(',') else line.split(',')
                    new_lines.append(line)
                anno = np.array(new_lines, dtype=np.float32)
                assert anno.shape[1] == 8, f'anno.shape: {anno.shape}'
                # VisDrone的标注格式：xmin, ymin, width, height, score, category, truncation, occlusion
                # score为0是评估中不考虑的
                if trte == 'test':
                    anno = anno[anno[:, 4] > 0]
                # category忽略区域不要
                anno = anno[anno[:, 5] != 0]
                # truncation不需处理
                # occlusion不需处理
                txt_content = ''
                for k, anno_ in enumerate(anno):
                    xmin, ymin, width, height, score, category, _, _ = anno_
                    xc = xmin + width / 2
                    yc = ymin + height / 2
                    xc, yc = xc / w, yc / h
                    bw, bh = width / w, height / h
                    txt_content += f'{int(category)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n'

                shutil.copy(f'{sp}/{sptr}/images/{img_name}', f'{tp}/images/{trte}/{img_name}')
                with open(f'{tp}/labels/{trte}/{img_name.replace(".jpg", ".txt")}', 'w') as f:
                    f.write(txt_content)

                show_str = f'[{i+1}/{len(trtes)}] [{j+1}/{len(imgs)}] {trte} {img_name} targets: {len(anno)}'
                # logger.info(f'\r{show_str.replace("\n", "")}')
                print(f'\r{show_str}', end='')
            print(f'\n{trte} done.')


if __name__ == '__main__':
    home = Path.home()
    argparser = argparse.ArgumentParser(description="VisDrone数据集转换为YOLO格式的数据集。")
    argparser.add_argument('--visdrone_path', type=str, help='VisDrone数据集路径。',
                           default=f'{home}/datasets/longhu/VisDrone/VisDrone2019-DET')
    argparser.add_argument('--yolo_path', type=str, help='YOLO格式的数据集路径。',
                           default=f'{home}/datasets/longhu/VisDrone/VisDrone2019-DET-YOLOfmt')
    argparser.add_argument('-f', '--force', action='store_true', help='强制覆盖已存在的目录。', default=False)
    opt = argparser.parse_args()

    # opt.force = True

    visdrone2yolo = VisDrone2YOLO(opt)
    visdrone2yolo.visdrone2yolo()
