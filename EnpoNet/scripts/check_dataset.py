import damei as dm
import os
from pathlib import Path
from collections import OrderedDict
from damei.data.check_dataset import CheckYOLO

pydir = Path(os.path.dirname(os.path.abspath(__file__)))


class CheckYOLO2(CheckYOLO):
    '''检查数据集每个类别共计多少个目标'''
    def __init__(self, dp=None):
        super().__init__(dp=dp)

    def __call__(self, trte=None, **kwargs):
        dp = self.dp
        trte = trte if trte else 'train'
        # print(dp)
        self.classes = kwargs.pop('classes', self.get_classes())
        info_dict = self.check(trte=trte)
        info_dict = dm.misc.dict2info(info_dict)
        print(f'{dp} {trte}\n{info_dict}')

    def check(self, trte):
        anno_dir = f'{self.dp}/labels/{trte}'
        labels = [f'{anno_dir}/{x}' for x in os.listdir(anno_dir) if x.endswith('.txt')]
        # print(f'labels: {len(labels)}')

        info_dict = OrderedDict()
        for _cls in self.classes:
            info_dict[_cls] = 0
        for label in labels:
            with open(label, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()  # 去掉\n
                if not line:
                    continue
                line = line.split()  # [0, x, y, w, h]
                cls_idx = line[0]
                cls = self.classes[int(cls_idx)]
                if cls in info_dict:
                    info_dict[cls] += 1
                else:
                    info_dict[cls] = 1
        return info_dict

    


def run():
    sps = f'/data/tml/newdata'
    for dataset in os.listdir(sps):
        sp = f'{sps}/{dataset}'
        for trte in ['train', 'test']:
            save_dir = f'{sp}/check/{trte}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            dm.data.check_YOLO(sp, trte, save_dir, show = False)

def run2():
    """
    检查YOLO格式每个类别多少张，得到dict
    """
    sps = f'/data/tml/mixed_polyp_v5_format'
    for dataset in os.listdir(sps):
        sp = f'{sps}/{dataset}'
        for trte in ['train', 'test']:
            cy = CheckYOLO2(dp=sp)
            cy(trte=trte)
    
if __name__ == '__main__':
    run()
    # run2()
    