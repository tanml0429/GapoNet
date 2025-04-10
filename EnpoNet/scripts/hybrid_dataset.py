"""
把多个yolov5数据集合并成1个
"""
import os, sys
from pathlib import Path
import shutil


class HybridYOLODatasets:

    def __init__(self) -> None:
        
        pass

    def __call__(self, sps, tp):

        # 创建文件夹        
        if not os.path.exists(f'{tp}'):
            os.makedirs(f'{tp}')

        for i, trte in enumerate(['train', 'test']):
            if not os.path.exists(f'{tp}/images/{trte}'):
                os.makedirs(f'{tp}/images/{trte}')
                os.makedirs(f'{tp}/labels/{trte}')
                # os.makedirs(f'{tp}/masks/{trte}')
            # 清空文件夹
            for root, dirs, files in os.walk(f'{tp}/images/{trte}'):
                for file in files:
                    os.remove(os.path.join(root, file))
            for root, dirs, files in os.walk(f'{tp}/labels/{trte}'):
                for file in files:
                    os.remove(os.path.join(root, file))

            last_stem = 0

            for j, sp in enumerate(sps):
                shutil.copy(f'{sp}/classes.txt', f'{tp}/classes.txt')
                stems = sorted([Path(x).stem for x in os.listdir(f'{sp}/images/{trte}') if x.endswith('.jpg') or x.endswith('.png') or x.endswith('.jpeg')])
                image_files = [x for x in  os.listdir(f'{sp}/images/{trte}') if x.endswith('.jpg') or x.endswith('.png') or x.endswith('.jpeg')]
                # label_files = [x for x in  os.listdir(f'{sp}/labels/{trte}') if x.endswith('.txt')]
                for k, image_file in enumerate(image_files):
                    stem = Path(image_file).stem
                    new_stem = f'{last_stem:0>8}_{stem}'
                    shutil.copy(f'{sp}/images/{trte}/{image_file}', f'{tp}/images/{trte}/{new_stem:0>6}.jpg')
                    shutil.copy(f'{sp}/labels/{trte}/{stem}.txt', f'{tp}/labels/{trte}/{new_stem:0>6}.txt')

                    print(f'\r[{i+1}/2] {trte} [{j+1}/{len(sps)}] {sp} [{k+1}/{len(stems)}] {stem} {new_stem}', end='')
                    last_stem += 1
    
                # for k, stem in enumerate(stems):
                # 复制图像
                #     new_stem = f'{last_stem:0>8}_{stem}'
                #     shutil.copy(f'{sp}/images/{trte}/{stem}.jpg', f'{tp}/images/{trte}/{new_stem:0>6}.jpg')
                #     shutil.copy(f'{sp}/images/{trte}/{stem}.jpeg', f'{tp}/images/{trte}/{new_stem:0>6}.jpeg')
                #     shutil.copy(f'{sp}/images/{trte}/{stem}.png', f'{tp}/images/{trte}/{new_stem:0>6}.png')
                #     shutil.copy(f'{sp}/labels/{trte}/{stem}.txt', f'{tp}/labels/{trte}/{new_stem:0>6}.txt')
                #     shutil.copy(f'{sp}/masks/{trte}/{stem}.npy', f'{tp}/masks/{trte}/{new_stem:0>6}.npy')

                #     print(f'\r[{i+1}/2] {trte} [{j+1}/{len(sps)}] {sp} [{k+1}/{len(stems)}] {stem} {new_stem}', end='')
                #     last_stem += 1
            print()

if __name__ == "__main__":
    data_root = f'/data/tml/mixed_polyp_v5_format'
    tp = f'/data/tml/hybrid_polyp_v5_format'

    sps = [f'{data_root}/{x}' for x in os.listdir(data_root)]

    hb = HybridYOLODatasets()
    hb(sps=sps, tp=tp)
