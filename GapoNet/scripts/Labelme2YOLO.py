"""
Labelme标注的数据转为YOLO格式
"""
import random
import collections
"""
VisDrone数据集转换为YOLO格式的数据集。
"""
import os, sys
from pathlib import Path
pydir = Path(os.path.abspath(__file__)).parent
import shutil
import argparse
import cv2

import numpy as np
import json
try:
    import damei as dm
except:
    sys.path.append(f'{pydir.parent.parent}/damei')
    import damei as dm
    sys.path.remove(f'{pydir.parent.parent}/damei')
logger = dm.getLogger('Labelme2YOLO')


class Labelme2YOLO(object):
    def __init__(self, opt):
        """
        # YOLO格式：cls, xc, yc, w, h. cls是类别的索引，xc, yc, w, h是分数坐标
        """
        self.opt = opt
        # self.classes = ["ignored regions", "pedestrian", "people", "bicycle", "car", "van",
        #                 "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"]
        self.classes = self.read_classes(opt.opt.sp)
        self.ratio = opt.ratio  # 训练集占比

    @staticmethod
    def read_classes(sp):
        # sp = self.opt.sp
        jsons = [f'{sp}/{f}' for f in os.listdir(sp) if f.endswith('.json')]

        classes_and_num = {}
        for j in jsons:
            with open(j, 'r') as f:
                d = json.load(f)
            shapes = d['shapes']
            for s in shapes:
                label = s['label']
                points = s['points']
                shape_type = s['shape_type']
                if label in classes_and_num.keys():
                    classes_and_num[label] += 1
                else:
                    classes_and_num[label] = 1
        classes = sorted(list(classes_and_num.keys()))
        print(f'Classes: {classes}')

        sorted_dict = collections.OrderedDict()
        for c in classes:
            sorted_dict[c] = classes_and_num[c]
        print(dm.misc.dict2info(sorted_dict))

        return classes

    def to_yolo(self):
        """
        将VisDrone数据集转换为YOLO格式的数据集。
        :param visdrone_path: VisDrone数据集路径。
        :param yolo_path: YOLO格式的数据集路径。
        :return: None
        """
        opt = self.opt
        sp = opt.sp
        tp = opt.tp

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
            f.writelines(['\n'.join(self.classes)] + ['\n'])

        # 将Labelme数据集转换为YOLO格式的数据集。
        trtes = ['train', 'test']
        jsons = [x for x in os.listdir(sp) if x.endswith('.json') and not x.startswith('.')]
        logger.info(f'{len(jsons)} labeled data (.json).')
        random.seed(930429)
        random.shuffle(jsons)

        train_json = jsons[:int(len(jsons) * self.ratio)]
        test_json = jsons[int(len(jsons) * self.ratio):]
        # print(len(jsons), jsons[-1], len(train_json), len(test_json))
        # exit()
        logger.info(f'{len(train_json)} train and {len(test_json)} will be transferred.')
        for i, trte in enumerate(trtes):
            # sptr = 'VisDrone2019-DET-train' if trte == 'train' else 'VisDrone2019-DET-val'

            trte_json = train_json if trte == 'train' else test_json
            for j, json_file in enumerate(trte_json):
                stem = Path(json_file).stem
                img_name = f'{stem}.jpg'
                # img_path = f'{sp}/{stem}_json/img.png'
                img_path = f'{sp}/{stem}.jpg'
                json_path = f'{sp}/{json_file}'
                assert os.path.exists(img_path), f'{img_path} not exists.'
                assert os.path.exists(json_path), f'{json_path} not exists.'

                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                with open(json_path, 'r') as f:
                    # lines = f.readlines()
                    annos = json.load(f)
                # print(annos.keys())

                # 验证宽高
                anno_h = annos['imageHeight']
                anno_w = annos['imageWidth']
                assert anno_h == h, f'{img_path} height not match. {anno_h} != {h}'
                assert anno_w == w, f'{img_path} width not match. {anno_w} != {w}'

                shapes = annos['shapes']
                txt_content = ''
                for k, shape in enumerate(shapes):
                    label = shape['label']
                    if label not in self.classes:
                        assert False, f'{label} not in {self.classes}'
                        # self.classes.append(label)
                    points = np.array(shape['points'])
                    shape_type = shape['shape_type']
                    if shape_type == 'rectangle' or shape_type == 'polygon':
                        xmin, ymin, xmax, ymax = points.min(axis=0).tolist()[0], points.min(axis=0).tolist()[1], \
                                                 points.max(axis=0).tolist()[0], points.max(axis=0).tolist()[1]
                        label_id = self.classes.index(label)
                        xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
                        bw, bh = xmax - xmin, ymax - ymin
                        xc, yc = xc / w, yc / h
                        bw, bh = bw / w, bh / h
                        # txt_content += f'{label_id} {xmin} {ymin} {xmax} {ymax}\n'
                        txt_content += f'{label_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n'
                    else:
                        raise NotImplementedError(f'{shape_type} not implemented.')

                # 图像和标注文件
                # shutil.copy(img_path, f'{tp}/images/{trte}/{img_name}')
                cv2.imwrite(f'{tp}/images/{trte}/{img_name}', img)
                with open(f'{tp}/labels/{trte}/{img_name.replace(".jpg", ".txt")}', 'w') as f:
                    f.write(txt_content)

                show_str = f'[{i+1}/{len(trtes)}]({trte}) [{j+1}/{len(trte_json)}] {trte} {img_name} targets:'
                # logger.info(f'\r{show_str.replace("\n", "")}')
                print(f'\r{show_str}', end='')
            print(f' {trte} done.')


if __name__ == '__main__':
    home = Path.home()
    argparser = argparse.ArgumentParser(description="Labelme数据集转换为YOLO格式的数据集。")
    argparser.add_argument('--sp', type=str, help='存放.json文件的路径',
                           default=f'{home}/datasets/longhu/examples/demo/anno_results_liqi/results')
    argparser.add_argument('--tp', type=str, help='输出结果保存的路径',
                           default=f'{home}/datasets/longhu/examples/demo/liqi_YOLOfmt')
    argparser.add_argument('--ratio', type=float, default=0.8, help='训练集占比，范围0~1，其余为测试集')
    argparser.add_argument('-f', '--force', action='store_true', help='强制覆盖已存在的目录。', default=False)
    opt = argparser.parse_args()

    opt.force = True
    opt.sp = f'{Path.home()}/datasets/longhu/datasets/jinglin/raw_80/merged'
    # opt.tp = f'{Path.home()}/datasets/longhu/datasets/jinglin/raw_80_augment_mss_YOLOfmt'
    opt.ratio = 1

    lb2yolo = Labelme2YOLO(opt)
    # lb2yolo.to_yolo()
