

import os, sys
import shutil
import tempfile

from ultralytics.utils.dist import find_free_network_port, generate_ddp_file
from ultralytics.utils.torch_utils import TORCH_1_9
from ultralytics.utils import USER_CONFIG_DIR

from GapoNet.configs import CONST

def generate_ddp_file_enpo(trainer):
    """Generates a DDP file and returns its file name."""
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)
    REPO_DIR = CONST.REPO_DIR

    content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
overrides = {vars(trainer.args)}

if __name__ == "__main__":
    import os, sys
    if '{REPO_DIR}' not in sys.path:
        sys.path.insert(0, '{REPO_DIR}')

    from GapoNet.apis import YOLO, LYMO
    from {module} import {name}
    # from ultralytics.utils import DEFAULT_CFG_DICT
    from GapoNet.apis.lymo_api import LYMO_DEFAULT_CFG_DICT

    # cfg = DEFAULT_CFG_DICT.copy()
    cfg = LYMO_DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)
    results = trainer.train()
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name

def generate_ddp_command_enpo(world_size, trainer):
    """Generates and returns command for distributed training."""
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/lightning/issues/15218

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    file = generate_ddp_file_enpo(trainer)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    
    cmd = [sys.executable, "-m", dist_cmd, "--nproc_per_node", f"{world_size}", "--master_port", f"{port}", file]
    return cmd, file

