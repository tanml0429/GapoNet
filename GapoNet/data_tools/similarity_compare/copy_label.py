import os, sys
from pathlib import Path
here = Path(__file__).parent

try:
    from GapoNet.version import __version__
except:
    sys.path.append(f'{here.parent.parent.parent}')
    from GapoNet.version import __version__

from GapoNet.configs import CONST
import shutil

def run():
    original_path = f'{CONST.DATASETS_DIR}/enpo_dataset/aug_10341'
    new_path = f'{CONST.DATASETS_DIR}/enpo_dataset/aug_61730'

    valid_suffix = [".png", ".jpg", ".jpeg"]
    img_names = [x for x in os.listdir(f"{new_path}/images") if Path(x).suffix in valid_suffix]

    for i, img_name in enumerate(img_names):
        # img_path = f'{new_path}/images/{img_name}'
        # source label path
        slp = f'{original_path}/labels/{img_name.replace(Path(img_name).suffix, ".txt")}'

        assert Path(slp).exists(), f"Label file not found: {slp}"

        # target label path
        tlp = f'{new_path}/labels/{img_name.replace(Path(img_name).suffix, ".txt")}'
        print(f"\r{i+1:0>5}/{len(img_names)}: {img_name} copied", end="", flush=True)
        if os.path.exists(tlp):
            # print(f"\n{tlp} already exists")
            continue
        shutil.copy(slp, tlp)
    print()
    pass
    
    

if __name__ == '__main__':
    run()
