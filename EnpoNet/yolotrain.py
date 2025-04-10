import os, sys
from pathlib import Path

from dataclasses import dataclass, field

from ultralytics import YOLO

import hepai as hai



def run(args):
    # Create a new YOLO model from scratch
    # model_name = args.pop('model')
    kwargs = args.__dict__
    model_name_or_cfg = kwargs.pop('model')
    model_weights = kwargs.pop('weights', None)
    model = YOLO(model_name_or_cfg, task=args.task)
    # model = YOLO(model_name_or_cfg, task=args.task)

    if model_weights:
        model = model.load(model_weights)
    
    # model = YOLO(model_name).load(model_weights)

    # freeze = kwargs.pop('freeze', '')
    # freeze = [x.strip() for x in freeze.split(',') if x]
    # kwargs['freeze'] = freeze

    results = model.train(**kwargs)

    # Evaluate the model's performance on the validation set
    results = model.val()
    print(results)

    # Perform object detection on an image using the model
    # results = model(f'{here}/lymonet/data/scripts/image.png')
    # print(results)

    # Export the model to ONNX format
    # success = model.export(format='onnx')

@dataclass
class Args:
    mode: str = 'train'
    model: str =  'yolo11n.pt'
    # weights: str = 'runs/detect/LYMO_MHSA_CA_RB14/weights/best.pt'
    data: str = '/home/tml/VSProjects/EnpoNet/EnpoNet/configs/data_cfgs/enpo_80.yaml'

    epochs: int = 300
    batch: int = 16
    imgsz: int = 640
    workers: int = 8
    device: str = '1'  # GPU id 
    # base_save_dir: str = '/data/tml/enpo_dataset/runs'
    name: str = '11n'
    patience: int = 0
    dropout: float = 0.5
    task: str = 'detect'
    amp: bool = True
    # freeze: str = '0'  # freeze layer 0,1,2,3 etc
    # freeze: str = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22"
    #box: float = 7.5  # 7.5 bbox loss gain  
    #cls: float = 0.5  # 0.5 cls loss gain
    #dfl: float = 1.5  # 1.5 dfl loss gain
    # content_loss_gain: float = 0.0
    # texture_loss_gain: float = 0.0
    # augment_in_training: bool = True  # Default True, Indicates whether to augment in training
    # load_correspondence: bool = False  # Default False, Indicates whether to load correspondence images, need `metadata.json` file in YOLO dataset
    
 
if __name__ == '__main__':
    args = hai.parse_args_into_dataclasses(Args)
    run(args)