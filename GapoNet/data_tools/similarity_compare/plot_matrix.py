import os
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from tqdm import tqdm
import sys
here = Path(__file__).parent
try:
    from GapoNet.version import __version__
except:
    sys.path.append(str(here.parent.parent.parent))
    from GapoNet.version import __version__
print(f"__version__: {__version__}")

from GapoNet.data_tools.similarity_compare.dataset_vector import ImageSimilarity
from GapoNet.configs import CONST



def down_sample(sm, kernel_size=3, stride=3, pool_method='max'):
    import torch
    sm = torch.tensor(sm).cuda()
    
    sm = sm.unsqueeze(0).unsqueeze(0)
    # m = torch.nn.AvgPool2d(5, stride=3)
    if pool_method == 'avg':
        m = torch.nn.AvgPool2d(kernel_size, stride=stride)
    elif pool_method == 'max':
        m = torch.nn.MaxPool2d(kernel_size, stride=stride)
    else:
        raise ValueError("pool_method must be 'avg' or 'max'")

    out = m(sm)
    return out.squeeze().cpu().numpy()

def run(matrix_path, img_dir_name, save_path=None):
    """Simular matrix plot."""
    # matrix_path = f"/data/tml/similar_matrix/hybrid_polyp_v5_format.npy"
    
    similarity_matrix = np.load(matrix_path)
    n = similarity_matrix.shape[0]
    sm = similarity_matrix

    # 手动降维s
    pool = False
    if pool:
        pool_method = 'max'
        pool_method = 'avg'
        sm = down_sample(sm, pool_method=pool_method)

    else:
        pool_method = 'none'

    for i in range(sm.shape[0]):
        sm[i, i] = 1
        sm[i, i+1::] = np.nan
        print(f"\r{i}", end="", flush=True)
    print()

    plt.imshow(sm, cmap='bwr',vmin=0, vmax=1)

    plt.colorbar()
    plt.savefig(f'{here}/results/{img_dir_name}_{pool_method}.png')
    print(f"Save plot to {here}/results/{img_dir_name}_{pool_method}.png")


def run_overall(imgs_path, dataset):
    """
    1. Images to vectors
    2. Similarity matrix
    3. Similarity matrix plot
    """
    from GapoNet.data_tools.similarity_compare.dataset_vector import run as run_images_to_vectors
    img_paths = run_images_to_vectors(imgs_path, dataset)

    from GapoNet.data_tools.similarity_compare.similar_matrix import run as run_vector_to_similarity_matrix
    matrix_path, sm = run_vector_to_similarity_matrix(dataset)

    # image_similarity = ImageSimilarity()
    # delete_list = image_similarity.delete_similar_images(sm, img_paths, target_path, threshold=0.9)


    

    # run(matrix_path, dataset)
    
    return sm, img_paths, matrix_path

if __name__ == "__main__":

    dataset = "aug_10341"
    imgs_path = f"{CONST.DATASET_DIR}/{dataset}/images"

    sm, img_paths, matrix_path = run_overall(imgs_path, dataset)
    

    # matrix_path = f"/data/tml/similar_matrix/{dataset}.npy"
    # img_dir_name = dataset
    # run(matrix_path, img_dir_name)
    # similarity_matrix = np.load(matrix_path)
    
    target_path = f"{CONST.DATASET_DIR}/filtered/{dataset}/images"
    image_similarity = ImageSimilarity()
    delete_list = image_similarity.delete_similar_images(sm, img_paths, target_path, threshold=0.9)
    
    # run(matrix_path, dataset)


    dataset_filtered = f'{dataset}_filtered'
    sm_filtered, img_paths_filtered, matrix_path_filtered = run_overall(target_path, dataset_filtered)
    plt.clf()
    # run(matrix_path_filtered, dataset_filtered)

    