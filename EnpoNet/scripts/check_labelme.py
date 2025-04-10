"""
检查label数据
"""
import os
from pathlib import Path
from Labelme2YOLO import Labelme2YOLO

sp = f'{Path.home()}/datasets/longhu/datasets/jinglin/raw_80/merged'
Labelme2YOLO.read_classes(sp=sp)
