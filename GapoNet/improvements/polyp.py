
import os
from copy import copy
from pathlib import Path
import torch
import subprocess

from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.dist import ddp_cleanup

from GapoNet.apis import YOLO
from GapoNet.apis.yolov8_api import (
    DEFAULT_CFG, DEFAULT_CFG_DICT, IterableSimpleNamespace,
    DetectionTrainer, DetectionValidator, DetectionPredictor, RANK,
    de_parallel, check_det_dataset, check_cls_dataset, clean_url, emojis
    )
from GapoNet.apis.lymo_api import LYMO_DEFAULT_CFG
from GapoNet.data_tools.utils import check_det_dataset as enpo_check_det_dataset
from GapoNet.improvements.utils.dist import generate_ddp_command_enpo

from .nn.tasks import LymoDetectionModel
from .val import LymoDetectionValidator
from .predict import LymoDetectionPredictor
from .dataset import LYMODataset, build_lymo_dataset
from .utils.utils import preprocess_correspondence
from .fine_cls_model.classification_validator import LymoClassificationValidator
from .fine_cls_model.classification_model import LymoClassificationModel
from .fine_cls_model.classify_trainer import LymoClassificationTrainer
from .utils.utils import update_save_dir


class LYMO(YOLO):
    
    def __init__(self, model: str | Path = 'yolov11n.pt', task=None) -> None:
        super().__init__(model, task)
    
    @property
    def task_map(self):
        task_map = super().task_map
        task_map['detect']['model'] = LymoDetectionModel
        task_map['detect']['trainer'] = LymoDetectionTrainer
        task_map['detect']['validator'] = LymoDetectionValidator
        task_map['detect']['predictor'] = LymoDetectionPredictor

        task_map["classify"]["model"] = LymoClassificationModel
        task_map["classify"]["trainer"] = LymoClassificationTrainer
        task_map["classify"]["validator"] = LymoClassificationValidator
        return task_map
    
    
    @staticmethod
    def apply_improvements():
        from .nn.nn import ResBlock_CBAM, CBAM, ChannelAttentionModule, SpatialAttentionModule, RecoveryBlock
        # globals().update(locals())
        globals()['RecoveryBlock'] = RecoveryBlock

        from GapoNet.apis.yolov8_api import BaseValidator
        # from ..ultralytics.ultralytics.engine import validator
        from .cfg import get_cfg
        # validator.get_cfg = get_cfg

        # print('apply_improvements')


class LymoDetectionTrainer(DetectionTrainer):

    def __init__(self, cfg=LYMO_DEFAULT_CFG, overrides=None, _callbacks=None):
        # self.args.save_dir = update_save_dir(self.args, overrides)
        # 把overrides的base_save_dir去掉，并且改成save_dir
        if overrides is not None:
            if 'base_save_dir' in overrides:
                base_save_dir = overrides.pop('base_save_dir')
                task = overrides.get('task')
                name = overrides.get('name')
                save_dir = f'{base_save_dir}/{task}/{name}'
                overrides['save_dir'] = save_dir
        super().__init__(cfg, overrides, _callbacks)
        

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = LymoDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights) 
        return model

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                # data = check_det_dataset(self.args.data)
                data = enpo_check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        self.data = data
        return data["train"], data.get("val") or data.get("test")

    
    def get_validator(self):
        # super().get_validator()
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'centent_loss', "texture_loss"
        return LymoDetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
    
    def build_dataset(self, img_path, mode='train', batch=None):
        """Build LYMO Dataset"""
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        # rect = mode == 'val'  # val时，rect=True
        rect = self.args.rect  # 在验证时，设置rect会报错

        return build_lymo_dataset(self.args, 
                                  img_path, batch, 
                                  self.data, mode=mode, 
                                  rect=rect,
                                  stride=gs)
    
    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch = super().preprocess_batch(batch)
        # batch = preprocess_correspondence(batch, self)
        if self.args.load_correspondence:
            cors = batch.get('correspondence', None) 
            assert len(cors) == len(batch['img']), "correspondence length should be equal to batch size"
            cors_img = torch.zeros_like(batch['img'])
            for i, cor in enumerate(cors):
                if cor is None:  # 没有对应的CDFI图像，就使用原图作为目标
                    cor_img = batch['img'][i]  
                else:
                    cor_img = cor['img']
                    cor_img = cor_img.to(self.device, non_blocking=True).float() / 255
                cors_img[i] = cor_img  # 保存对应的CDFI图像
            # 连续
            cors_img = cors_img.to(self.device, non_blocking=True).float()
            batch['cors_img'] = cors_img
        else:
            batch['cors_img'] = None
        
        return batch
    
    def train(self):
        """
        更新GapoNet的train方法，适配多卡训练
        """
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0
            
        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch == -1:
                LOGGER.warning(
                    "WARNING ⚠️ 'batch=-1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.args.batch = 16

            # Command
            # cmd, file = generate_ddp_command(world_size, self)
            cmd, file = generate_ddp_command_enpo(world_size, self)
            try:
                LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train(world_size)
        pass

