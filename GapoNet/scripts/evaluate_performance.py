"""
对Json格式的真值和预测值，采用混淆矩阵方法评估性功能
"""
import os
from pathlib import Path
import numpy as np
import json
# import damei as dm
from dmapi import dm

import Labelme2YOLO

logger = dm.getLogger('evaluate_performance.py')


class EvaluatePerformance(object):
    def __init__(self, gt, pd, iou_thresh=0.65, merge=True):
        self.gt = gt
        self.pd = pd
        self.classes = Labelme2YOLO.Labelme2YOLO.read_classes(self.gt)
        self.iou_thresh = iou_thresh
        self.merge = merge

    def __call__(self, *args, **kwargs):
        gt_jsons = sorted([f'{self.gt}/{x}' for x in os.listdir(self.gt) if x.endswith('.json')])
        pd_jsons = sorted([f'{self.pd}/{x}' for x in os.listdir(self.pd) if x.endswith('.json')])
        assert len(gt_jsons) == len(pd_jsons), 'GT and PD json number not equal'

        cm = np.zeros((len(self.classes)+1, len(self.classes)+1), dtype=np.int32)
        for i, gt_json in enumerate(gt_jsons):  # 遍历json，gt和pd一一对应
            stem = Path(gt_json).stem
            pd_json = f'{self.pd}/{stem}.json'
            assert os.path.exists(pd_json), f'{pd_json} not exists'
            gt_shapes = self.read_shapes(gt_json)
            pd_shapes = self.read_shapes(pd_json)
            # print(len(pd_shapes), len(gt_shapes))
            for j, gt_shape in enumerate(gt_shapes):
                gt_class = gt_shape['label']
                gt_class_id = self.classes.index(gt_class)
                matched_pd_class_id = self.match_shape(gt_shape, pd_shapes)
                if matched_pd_class_id is None:  # 未匹配到，漏检
                    # print(f'{gt_class} not matched {j}')
                    cm[gt_class_id, -1] += 1
                elif matched_pd_class_id == gt_class_id:  # 正确
                    # print(f'matched {gt_class} {j}')
                    cm[gt_class_id, gt_class_id] += 1
                else:  # 匹配到，错检
                    # print(f'wrong detect {gt_class} != {self.classes[matched_pd_class_id]} {j}')
                    cm[gt_class_id, matched_pd_class_id] += 1
            # 误检(gt没有，pd多出来的)
            for k, pd_shape in enumerate(pd_shapes):
                pd_class = pd_shape['label']
                pd_class_id = self.classes.index(pd_class)  # pd的id
                cm[-1, pd_class_id] += 1
            print(f'\r{i+1}/{len(gt_jsons)} {stem}', end='')

        # 打印一下
        cm_list = np.array(cm)
        first_col = np.array([]+self.classes+['background']).reshape(-1, 1)
        cm_list = np.concatenate((first_col, cm_list), axis=1).tolist()
        cm_list.insert(0, ['']+[self.classes]+['background'])
        print(f'\nConfusion matrix:\n{dm.misc.list2table(cm_list, alignment=">")}')

        # 计算分数
        score = dm.general.confusion2score(
            cm, round=4, with_background=True, return_type='list', percent=False,
            names=self.classes)
        # print(score)
        # print(f'score:\n{dm.misc.dict2info(score)}')
        print(f'Scores: \n{dm.misc.list2table(score, float_bit=4, alignment=">")}')
        return score

    def match_shape(self, gt_shape, pd_shapes):
        gt_points = gt_shape['points']
        gt_label = gt_shape['label']
        gt_bbox = dm.general.pts2bbox(gt_points)
        if len(pd_shapes) == 0:
            logger.info(f'mathch "{gt_label}" failed, no pd shape')
            return None
        ious = []
        for i, pd_shape in enumerate(pd_shapes):
            pd_points = pd_shape['points']
            pd_bbox = dm.general.pts2bbox(pd_points)
            iou = dm.general.bbox_iou(gt_bbox, pd_bbox, x1y1x2y2=True, return_np=True)
            ious.append(iou)
        ious = np.array(ious)
        # print(ious)
        max_iou = np.max(ious)
        max_iou_id = np.argmax(ious)
        if max_iou >= self.iou_thresh:
            matched_pd_shape = pd_shapes.pop(max_iou_id)
            pd_class = matched_pd_shape['label']
            pd_class_id = self.classes.index(pd_class)
            # np.delete(pd_shapes, max_iou_id)  # 删除匹配上的pd_shape
            return pd_class_id
        return None

    def read_shapes(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        # print(data, data.keys())
        # exit()
        shapes = data['shapes']
        return shapes


if __name__ == '__main__':
    gt = f'{Path.home()}/datasets/longhu/datasets/jinglin/raw_80'
    pd = f'{Path.home()}/datasets/longhu/datasets/jinglin/pred_raw_80'
    pd = f'{Path.home()}/datasets/longhu/datasets/jinglin/raw_80/merged'
    ep = EvaluatePerformance(gt, pd, merge=True)
    ep()
