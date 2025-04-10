"""
在进行数据增强之前，把李琦标注的数据格式转换成labelme格式
李琦的格式：
/root/
    |- stem.json (json文件)
    |- stem_json (文件夹)
        |- img.png (原始图像)
        |- xxx
labelme的格式：
/root/
    |- stem.json (json文件)
    |- stem.png (图像)
"""

import os, sys
from pathlib import Path
pydir = Path(os.path.abspath(__file__)).parent
import shutil
import cv2
import json

try:
    import damei as dm
except:
    sys.path.append(f'{pydir.parent.parent}/damei')
    import damei as dm

logger = dm.get_logger('liqi2labelme')


def run():
    liqi_path = f'{Path.home()}/datasets/longhu/examples/demo/tianjin_fine_anno/json'
    tp = f'{Path.home()}/datasets/longhu/examples/demo/tianjin_fine_labelme'

    json_files = sorted([x for x in os.listdir(liqi_path) if x.endswith('.json')])
    logger.info(f'{len(json_files)} json files found.')
    if os.path.exists(tp):
        shutil.rmtree(tp)
    os.makedirs(tp)

    for i, json_file in enumerate(json_files):
        stem = Path(json_file).stem
        new_stem = f'{i:>06d}'
        shutil.copy(f'{liqi_path}/{json_file}', f'{tp}/{new_stem}.json')

        # shutil.copy(f'{liqi_path}/{stem}_json/img.png', f'{tp}/{new_stem}.png')
        img = cv2.imread(f'{liqi_path}/{stem}_json/img.png')
        cv2.imwrite(f'{tp}/{new_stem}.jpg', img)

        logger.info(f'{i+1}/{len(json_files)}: {json_file} -> {new_stem}.json')


if __name__ == '__main__':
    run()
