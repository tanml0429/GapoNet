

import os, sys
from pathlib import Path
here = Path(__file__).parent
import numpy as np

try:
    from EnpoNet.version import __version__
except:
    sys.path.append(str(here.parent.parent.parent))
    from EnpoNet.version import __version__
print(f"__version__: {__version__}")

from EnpoNet.data_tools.similarity_compare.dataset_vector import ImageSimilarity
from EnpoNet.configs import CONST


def run(dataset_name):
    img_sim = ImageSimilarity()


    vectors_dir = f'{CONST.DATASET_DIR}/mixed_polyp_vector/{dataset_name}'
    npy_list = [f'{vectors_dir}/{x}' for x in os.listdir(vectors_dir) if x.endswith(".npy")]
    vector_array = np.concatenate([np.load(x) for x in npy_list], axis=0)
    # vector_array = np.squeeze(vector_array)
    print(vector_array.shape)  # (n, 768), i.e. (73856, 768)

    save_path = f'{CONST.DATASET_DIR}/similar_matrix/{dataset_name}.npy' # /data/tml/similar_matrix/hybrid_polyp_v5_format.npy
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sm = img_sim.similarity_matrix(
        vectors=vector_array,
        save_path=save_path,
        use_cuda=True,
        n_parts=1,
    )
    return save_path, sm

if __name__ == "__main__":
    
    run()
