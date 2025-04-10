# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This script is used to calculate the high-dimensional features of images  #
# Author: Menglu Tan                                                        #
# Date: 2024-03-01                                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


from typing import List, Dict, Literal
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
from EnpoNet.configs import CONST
here = Path(__file__).parent


class ImageDataset(Dataset):
    """用于加载图像数据集和前处理"""
    
    def __init__(
            self, 
            img_paths: List[str], 
            model_name: Literal["facebook/dinov2-base"] = "facebook/dinov2-base", 
            transform=None,
            verbose: bool = False
            ):
        self.img_paths = img_paths
        self.transform = transform

        self.model_name = model_name
        self._processor = None
        self.verbose = verbose
        
    @property
    def processor(self):
        if self._processor is None:
            if self.verbose:
                print("Loading processor")
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        return self._processor

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # img = read_image(img_path)
       
        img = self.processor(images=Image.open(img_path), return_tensors="pt")
        img_tensor = img['pixel_values'][0, :, :, :]
        # 变回字典
        img = {'pixel_values': img_tensor}
        if self.transform:
            img = self.transform(img)

        return img


class ImageSimilarity:
    """
    使用DINOv2模型计算图像的高维特征，以此计算数据集中所有数据的相似度。
    一方面可用于评估数据集质量，另一方面可用于数据集去重。
    """

    def __init__(self, device: str = "cuda:0"):
        self._model = None
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
        pass
        

    @property
    def model(self):
        if self._model is None:
            # print("Loading model")
            self._model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
            # print("Model loaded")
        return self._model    

    def get_features(self, images):
        """
        Input pre-processed images, return the features.
        """
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs
    
    def cosine_similarity_back(self, x, y):
        """calculate cosine similarity between x and y"""
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    
    def run(self, imgs):
        
        n = len(imgs)
        # output = self.get_features(imgs)
        inputs = {k: v.to(self.device) for k, v in imgs.items()} # 将数据放到GPU上
        output = self.model(**inputs)
        vectors = output.pooler_output.detach().cpu().numpy()  # [n, 768]
        return vectors

    def cosine_similarity_jit(self, v1: np.array, v2: np.array):
        """calculate cosine similarity between v1 and v2 using matrix multiplication"""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)


    def similarity_matrix(
            self, 
            vectors: np.ndarray,
            vectors2: np.ndarray = None,
            save_path: str = None, 
            dataset_name: str = None, 
            use_cuda: bool = True,
            use_ergodic: bool = False,
            device: int = 0,
            n_parts: int =1,
            ):
        """
        Calculate the similarity matrix of vectors.
        :param vectors: np.ndarray, shape (n, 768)
        :param save_path: str, path to save the similarity matrix
        :param dataset: ImageDataset, used to get the number of images
        :param use_cuda: bool, use cuda or not
        :param use_ergodic: bool, use ergodic or not
        :param device: int, GPU device
        :param n_parts: int, number of parts to split the similarity matrix when calculating
        """
        if save_path is not None and os.path.exists(save_path):
            # print(f"Similarity matrix already exists in `{save_path}`")
            return np.load(save_path)

        n = len(vectors)  # number of vectors
        m = len(vectors2) if vectors2 is not None else n
        pbar = tqdm(total=n)
        if use_ergodic:  # 最慢的方式，遍历所有向量
            for i in range(n):
                for j in range(i, n):
                    sml = self.cosine_similarity_jit(vectors[i], vectors[j])
                    similarity_matrix[i, j] = sml
                    similarity_matrix[j, i] = sml
                pbar.update(1)
        elif not use_cuda:  # 使用矩阵乘法和多核CPU
            similarity_matrix = np.zeros((n, n), dtype=np.float16)
            for i in range(n):
                v1 = vectors[i]  # (768,)
                v2 = vectors.T  # (n, 768) -> (768, n)
                dot_product = np.dot(v1, v2)  # (n,)
                norm_v1 = np.linalg.norm(v1)  # (1, )
                norm_v2 = np.linalg.norm(v2, axis=0)  # (n,)
                res = dot_product / (norm_v1 * norm_v2)  # (n,)
                similarity_matrix[i] = res
                pbar.update(1)
        else:  # 使用GPU
            torch.cuda.set_device(device)
            # torch.cuda.set_device(self.device)
            vectors = torch.tensor(vectors).cuda()
            if vectors2 is not None:
                vectors2 = torch.tensor(vectors2).cuda()
            
            np_sim_m = np.zeros((n, m), dtype=np.float16)
            chunk_size = n // n_parts  # 一块卡装不下时，分chunk计算
            for i in range(0, n, chunk_size):
                # similarity_matrix = torch.zeros((n, m), dtype=torch.float16).cuda()
                similarity_matrix = torch.zeros((chunk_size, m), dtype=torch.float16).cuda()
                start_idx, end_idx = i, min(i+chunk_size, n)
                print(f"Processing chunk from index {start_idx} to {end_idx}")
                # for i in range(n):
                for j, idx in enumerate(range(start_idx, end_idx)):
                    v1 = vectors[idx].unsqueeze(0)
                    v2 = vectors.T if vectors2 is None else vectors2.T
                    # dot_product = torch.dot(v1, v2)
                    dot_product = torch.matmul(v1, v2)
                    norm_v1 = torch.norm(v1)
                    norm_v2 = torch.norm(v2, dim=0)
                    res = dot_product / (norm_v1 * norm_v2)  # 
                    # assert torch.min(res) >= 0.0, f"min: {torch.min(res)}"
                    similarity_matrix[j] = res
                    pbar.update(1)
                similarity_matrix = similarity_matrix.detach().cpu().numpy()
                np_sim_m[start_idx:end_idx] = similarity_matrix        
        pbar.close()

        if save_path is None:
            save_path = f'{here}/results/{dataset_name}_matrix.npy'
        # similarity_matrix = np.array(similarity_matrix, dtype=np.float16)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, similarity_matrix)
        print(f"Chunk result from index {start_idx} to {end_idx}:")
        print(f"Simularity matrix: {similarity_matrix}")
        return similarity_matrix

    def delete_similar_images(self, 
            similarity_matrix, 
            imgs, 
            target_folder, 
            threshold=0.9):
        
        delete_list = self.cal_delete_list(similarity_matrix, threshold)

        return self.delete_images_by_delete_list(imgs, delete_list, target_folder)

    def cal_delete_list(
            self,
            similarity_matrix: np.ndarray,
            threshold: float = 0.9,
            exclude_digonal: bool = None,
            only_upper_triangle: bool = True
        )-> List[int]:
        """
        Calculate the index of images to be deleted, return a list of indices.
        :param similarity_matrix: np.ndarray, shape (n, n)
        :param threshold: float, similarity threshold
        :param exclude_digonal: bool, exclude the diagonal or not
        """

        n, m = similarity_matrix.shape
        if n != m:
            exclude_digonal = False
            only_upper_triangle = False
        else:
            exclude_digonal = True
            
        n = similarity_matrix.shape[0]
        delete_list = []
        for i in range(n):
            if i+1 == n and exclude_digonal:
                continue
            data = similarity_matrix[i, i+1::] if only_upper_triangle else similarity_matrix[i]
            max_sm = np.max(data)
            if max_sm > threshold:
                delete_list.append(i)
            # print(f"\rImage {i}", end="", flush=True)
        # print()
        return delete_list
    
    def delete_images_by_delete_list(self, imgs, delete_list, target_folder):
        keep_list = [img for idx, img in enumerate(imgs) if idx not in delete_list]
        if os.path.exists(target_folder):
            shutil.rmtree(target_folder)
            os.makedirs(target_folder, exist_ok=True)
        else:
            os.makedirs(target_folder, exist_ok=True)
        # 文件名重复时不覆盖
        print(f"Original images: {len(imgs)}")
        print(f"Filtered images: {len(keep_list)}") 
        for i in keep_list:
            filename = i.split("/")[-1]
            target_file_path = f"{target_folder}/{filename}"
            if os.path.exists(target_file_path):
                # 上一级文件夹名
                parent_folder = i.split("/")[-2]
                filename = f"{parent_folder}_{filename}"
                target_file_path = f"{target_folder}/{filename}"
            shutil.copy(i, target_file_path)

        print(f"Filtered images saved to `{target_folder}`")  
        pass


    def get_img_paths(self, root_dir: str) -> List[str]:
        """Get all image paths in the root and its subdirectories."""
        img_files = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if Path(file).suffix in ['.jpg', '.png', '.jpeg']:
                    img_files.append(os.path.join(root, file))
        img_files = sorted(img_files)
        print(f"Number of images: {len(img_files)}")
        img_paths = img_files
        return img_paths

    
    def data2vector(
            self, 
            dataset: ImageDataset, 
            # dataset_name: str, 
            output_dir: str, 
            batch_size: int=64,
            **kwargs
        )-> str:
        """
        Calculate the vectors of all images in the dataset and save them to the output_dir.
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        vector_list = []
        # vector_output_dir = f"/data/tml/mixed_polyp_vector/{dataset_name}"
        vector_output_dir = output_dir
        if os.path.exists(vector_output_dir):
            vector_files = os.listdir(vector_output_dir)
            if len(vector_files) > 0:
                # print(f"Vector files already exist in `{vector_output_dir}`")
                return vector_output_dir
        else:
            os.makedirs(vector_output_dir, exist_ok=True)
        # 进度条
        for i, batch in enumerate(tqdm(dataloader)):
            imgs = batch
            vectors = self.run(imgs) # (bs, 768)
            vector_list.extend(vectors)

            if (i+1) % 100 == 0:
                vector_array = np.vstack(vector_list)
                np.save(f'{vector_output_dir}/vector_{i+1}', vector_array)
                vector_list = []

        if vector_list:
            vector_array = np.vstack(vector_list)
            np.save(f'{vector_output_dir}/vector_{i+1}', vector_array)

        print(f"All vectors saved to `{vector_output_dir}`")
        return vector_output_dir


    def load_vector(self, vector_dir: str) -> np.ndarray:
        npy_list = [f'{vector_dir}/{x}' for x in os.listdir(vector_dir) if x.endswith(".npy")]
        if len(npy_list) == 0:
            raise FileNotFoundError(f"No vector files found in `{vector_dir}`")
        vector_array = np.concatenate(
            [np.load(x) for x in npy_list], axis=0)
        return vector_array

def run(imgs_path, dataset_name):
    
    image_similarity = ImageSimilarity()
    img_paths = image_similarity.get_img_paths(imgs_path)
    data = ImageDataset(img_paths, 'facebook/dinov2-base')
    output_dir = f'{CONST.DATASET_DIR}/mixed_polyp_vector/{dataset_name}'
    image_similarity.data2vector(data, output_dir=output_dir, batch_size=64)

    return img_paths
    

if __name__ == "__main__":
    
    dataset = "CVC-ColonDB"
    # dataset2 = "train2019"
    imgs_path = f"/data/tml/mixed_polyp/{dataset}/images"
    # img_path = f"/data/tml/mixed_polyp/{dataset}/images"
    # img_path = f"{here}"

    run(imgs_path, dataset) 
    
    


