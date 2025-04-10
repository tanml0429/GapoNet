"""
基于multi scale slice的数据集增强算法
"""

import shutil
import os, sys
from pathlib import Path
pydir = Path(os.path.abspath(__file__)).parent

try:
    import damei as dm
except:
    sys.path.append(f'{pydir.parent.parent}/damei')
    import damei as dm


def run():
    dataset_dir = "/home/zzd/datasets/longhu/datasets/jinglin"

    tp = f'{dataset_dir}/raw_80_augment_mss'
    if os.path.exists(tp):
        shutil.rmtree(tp)

    dm.data.augment_mss(
        source_path=f'{dataset_dir}/raw_80',
        target_path=tp,
        min_wh=800,  # 默认640
        max_wh=None,
        stride_ratio=0.75,  # 默认0.5
        pss_factor=1.25,  # 默认1.25
        need_annotation=True,  # 默认False
        anno_fmt='json',
    )


if __name__ == '__main__':
    run()