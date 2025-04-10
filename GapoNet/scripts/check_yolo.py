import os, sys
from pathlib import Path
pydir = Path(os.path.abspath(__file__)).parent
try:
    import damei as dm
except:
    sys.path.append(f'{pydir.parent.parent}/damei')
    import damei as dm
    sys.path.remove(f'{pydir.parent.parent}/damei')


# dp = f'{Path.home()}/datasets/longhu/VisDrone/VisDrone2019-DET-YOLOfmt'
dp = f'{Path.home()}/datasets/longhu/examples/demo/liqi_YOLOfmt'
# dp = f'{Path.home()}/datasets/longhu/examples/demo/tianjin_fine_labelme_augmented_YOLOfmt'
dp = f'{Path.home()}/datasets/longhu/examples/demo/tianjin_fine_labelme_augmented_YOLOfmt4000'
dp = f'{Path.home()}/datasets/longhu/examples/demo/tianjin_fine_hybrid_YOLOfmt4050'
dp = f'{Path.home()}/datasets/longhu/examples/demo/tianjin_fine_YOLOfmt'
dp = f'{Path.home()}/datasets/longhu/datasets/jinglin/augment_YOLOfmt_8000'
dp = f'{Path.home()}/datasets/longhu/datasets/jinglin/raw_80_YOLOfmt'
# dp = f'{Path.home()}/datasets/longhu/datasets/jinglin/raw_80_augment_mss_YOLOfmt'
dp = '/home/tml/datasets/enpo_dataset/mixed_69959'

save_dir = f'{Path(dp).parent}/check'

# save_dir = '.'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
dm.data.check_YOLO(dp, trte='test', save_dir=save_dir, show=False)
