import os, sys
from pathlib import Path
here = Path(__file__).parent


REPO_DIR = f'{here.parent.parent}'
DATASETS_DIR = '/home/tml/datasets'
DATASET_DIR = f'{DATASETS_DIR}/enpo_dataset'

# CONFIGS
CONFIGS_DIR = f'{REPO_DIR}/EnpoNet/configs'
MODEL_CFG_DIR = f'{REPO_DIR}/EnpoNet/configs/model_cfgs'
DATA_CFG_DIR = f'{REPO_DIR}/EnpoNet/configs/data_cfgs'




