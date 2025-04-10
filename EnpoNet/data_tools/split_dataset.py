# # # # # # # # # # # # # # # # # # # # # # # # #  # # # # # # # # # # # # # #  # # # # # # # # # # # # # # 
# 该脚本用于构建存在数据泄露和不存在数据泄露的两个数据集，用于进行数据泄露消融实验，说明其对模型性能的影响。
# 需准备两个数据集：（1）原始Mixed数据集73856张，（2）EndoPolyp-13182数据集（由73856自去重后构建的数据集，未划分训练测试和验证）。
# 输出：
# 1. 将EndoPolyp-13182数据集划分为训练、验证和测试集，数据无泄漏。
# 2. 由Mixed-73856数据集构建Mixed-45327数据集，其中，
#  - 训练集：（Mixed_73856 - EndoPolyp_test_1319 - 与EndoPolyp_test_1319相似的图片）* 90%
#  - 验证集：（Mixed_73856 - EndoPolyp_test_1319 - 与EndoPolyp_test_1319相似的图片）* 10%
#  - 测试集：与EndoPolyp相同，即EndoPolyp_test_1319
# 特点：
#  - Mixed-45327数据集中训练集与验证集存在数据泄露，训练集和测试集无数据泄露。
#  - EndoPolyp-13182数据集中训练集、验证集和测试集均无数据泄露。
# Author: Menglu Tan
# Date: 2024-05-24
# # # # # # # # # # # # # # # # # # # # # # # # #  # # # # # # # # # # # # # #  # # # # # # # # # # # # # #

from typing import List, Iterable, Literal
import os, sys
from pathlib import Path
here = Path(__file__).parent
import random
import shutil
import numpy as np

from dataclasses import dataclass
import hepai as hai

try:
    from EnpoNet.version import __version__
except:
    sys.path.append(f'{here.parent.parent}')
    from EnpoNet.version import __version__
from EnpoNet.configs import CONST
from EnpoNet.data_tools import ImageDataset, ImageSimilarity


class MixedPolypSplitor:

    def __init__(self, args) -> None:
        self.args = args
        self.enpo_13000_dir_org = f'{CONST.DATASETS_DIR}/enpo_dataset/enpo_12927_origin'
        self.enpo_73000_dir_org = f'{CONST.DATASETS_DIR}/enpo_dataset/aug_61730'



    def copy_file(
            self, 
            imgs: Iterable[str], 
            tvt: Literal["train", "val", "test"],
            target_dir: str, 
            source_dir: str = None,
            auto_source_dir: bool = True,
            extra_dir: str = None
            ):
        """tvt: train or val or test"""

        if not os.path.exists(f'{target_dir}/images/{tvt}'):
            os.makedirs(f'{target_dir}/images/{tvt}', exist_ok=True)
        if not os.path.exists(f'{target_dir}/labels/{tvt}'):
            os.makedirs(f'{target_dir}/labels/{tvt}', exist_ok=True)

        in_flag = False
        num_bit = len(str(len(imgs)))
        for i, img in enumerate(imgs):
            # 判断是不是绝对路径
            if os.path.exists(img) and auto_source_dir:
                source_dir = Path(img).parent.parent
                img = Path(img).name
            if extra_dir is None:
                img_path = f'{source_dir}/images/{img}'
                label_path = f'{source_dir}/labels/{img.replace(Path(img).suffix, ".txt")}'
            else:
                img_path = f'{source_dir}/images/{extra_dir}/{img}'
                label_path = f'{source_dir}/labels/{extra_dir}/{img.replace(Path(img).suffix, ".txt")}'
            
            target_path = f'{target_dir}/images/{tvt}/{img}'
            if os.path.exists(target_path):
                continue
            in_flag = True
            print(f"\r{i+1:0>{num_bit}}/{len(imgs)}: {img} copied", end="", flush=True)
            shutil.copy(img_path, target_path)
            lb_target_path = f'{target_dir}/labels/{tvt}/{img.replace(Path(img).suffix, ".txt")}'
            shutil.copy(label_path, lb_target_path)
        if in_flag:
            print()

    def split_13000_into_train_val_test(self):
        """Split the 13000 dataset into train, val and test"""
        ratio = [0.8, 0.1, 0.1]

        valid_suffix = [".png", ".jpg", ".jpeg"]
        imgs = [x for x in os.listdir(f"{self.enpo_13000_dir_org}/images") if Path(x).suffix in valid_suffix]
    
        random.seed(429)
        random.shuffle(imgs)

        train, val, test = np.split(
            imgs, 
            [int(len(imgs)*ratio[0]), 
             int(len(imgs)*(ratio[0]+ratio[1]))])

        num_imgs = len(train) + len(val) + len(test)
        s_dir = self.enpo_13000_dir_org 
        o_dir = f'{CONST.DATASETS_DIR}/enpo_dataset/enpo_{num_imgs}'
        self.copy_file(train, tvt="train", target_dir=f'{o_dir}', source_dir=s_dir)
        self.copy_file(val, tvt="val", target_dir=f'{o_dir}', source_dir=s_dir)
        self.copy_file(test, tvt="test", target_dir=f'{o_dir}', source_dir=s_dir)
        
        print(f"Total: {num_imgs}, Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

        return o_dir

    def split_73000_into_train_val_test(self, enpo_dir: str):
        """
        Split the 73000 dataset into train, val and test
        注意：
        测试集：与13000的测试集相同
        训练集: (73000-1300的训练集-与1300的训练集相似的图片) * 90%
        验证集：(73000-1300的训练集-与1300的训练集相似的图片) * 10%
        """
        
        sim_calc = ImageSimilarity(device=self.args.device)

        # 计算原始的73000数据集的特征项目，并保存到硬盘
        mixed73000_img_dir = f"{self.enpo_73000_dir_org}/images"
        mixed73000_img_paths = self.list_image_paths(mixed73000_img_dir, sort=True)
        num_imgs = len(mixed73000_img_paths)
        vector73000_dir = f"{CONST.DATASETS_DIR}/enpo_dataset/vectors/mixed_{num_imgs}"
        self.caculate_vector(sim_calc, mixed73000_img_paths, vector73000_dir)
        
        # 计算enpo测试集1300的特征项目，并保存到硬盘
        enpo_test_img_dir = f"{enpo_dir}/images/test"
        enpo_test_img_paths = self.list_image_paths(enpo_test_img_dir, sort=True)
        num_imgs_test = len(enpo_test_img_paths)
        vector_enpo_test_dir = f"{CONST.DATASETS_DIR}/enpo_dataset/vectors/enpo_test_{num_imgs_test}"
        self.caculate_vector(sim_calc, enpo_test_img_paths, vector_enpo_test_dir)

        # 计算73000与1300测试集的相似度矩阵
        vec73000 = sim_calc.load_vector(vector73000_dir)
        vec_enpo_test = sim_calc.load_vector(vector_enpo_test_dir)
        matrix_path = f"{CONST.DATASETS_DIR}/enpo_dataset/sim_matrices/matrix_{num_imgs}vs{num_imgs_test}.npy"
        sim_matrix = sim_calc.similarity_matrix(
            vec73000, vectors2=vec_enpo_test,
            save_path=matrix_path,
            use_cuda=True
            )
        
        # 根据相似度矩阵，构建新的数据集
        delete_list = sim_calc.cal_delete_list(sim_matrix, threshold=0.9, exclude_digonal=False)
        print(f"原始73000中与13000的测试集1319张的重复率大于0.9的图片数量：{len(delete_list)}，\n重复比例：{len(delete_list)/len(sim_matrix)}")
        valid_img_paths = self.get_valid_img_paths(mixed73000_img_paths, delete_list)
        train, val = self.split_train_val(valid_img_paths, ratio=[1, 0])
        test_s_dir = enpo_test_img_paths[0].split("/images")[0]
        test = [x.split("/images/test/")[1] for x in enpo_test_img_paths]


        num_imgs = len(train) + len(val) + len(test)
        o_dir = f'{CONST.DATASETS_DIR}/enpo_dataset/mixed_{num_imgs}'
        self.copy_file(train, tvt='train', target_dir=o_dir)
        self.copy_file(val, tvt='val', target_dir=o_dir)
        self.copy_file(
            test, tvt='test', 
            target_dir=o_dir,
            source_dir=test_s_dir, auto_source_dir=False, extra_dir="test")

        print(f"Total: {num_imgs}, Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        pass

    

    def split_train_val(self, img_paths: List[str], ratio: List[float]):
        random.seed(429)
        random.shuffle(img_paths)
        train, val = np.split(
            img_paths, 
            [int(len(img_paths)*ratio[0])])

        return train, val

    def get_valid_img_paths(self, img_paths: List[str], delete_list: List[int]):
        imgs = list()
        for i in range(len(img_paths)):
            if i in delete_list:
                continue
            imgs.append(img_paths[i])
        # for i, idx in enumerate(delete_list):
        #     imgs.append(img_paths[idx])
        return imgs
        
    def list_image_paths(self, img_dir: str, sort=False):
        valid_suffix = [".png", ".jpg", ".jpeg"]
        imgs = [x for x in os.listdir(img_dir) 
                if Path(x).suffix in valid_suffix]
        img_paths = [f"{img_dir}/{img}" for img in imgs]
        if sort:
            img_paths = sorted(img_paths)
        return img_paths

    def caculate_vector(
            self, sim_calc: ImageSimilarity, 
            img_dir_or_image_paths: str | List,
            output_dir: str,
            **kwargs,
            ):
        
        if isinstance(img_dir_or_image_paths, list):
            img_paths = img_dir_or_image_paths
        else:
            img_paths = self.list_image_paths(img_dir_or_image_paths)
        dataset = ImageDataset(img_paths)
        return sim_calc.data2vector(
            dataset, 
            output_dir=output_dir,
            batch_size=64,
            **kwargs
            )
       
    def calculate_mix73000_self_repeat_rate(self):
        sim_calc = ImageSimilarity()
        vector73000_dir = f"{CONST.DATASETS_DIR}/enpo_dataset/vectors/mixed_64635"
        vec73000 = sim_calc.load_vector(vector73000_dir)
        matrix_path = f"{CONST.DATASETS_DIR}/enpo_dataset/matrices/matrix_64635vs64635.npy"
        sim_matrix = sim_calc.similarity_matrix(
            vec73000,
            # vectors2=vec_enpo_test,
            save_path=matrix_path,
            use_cuda=True,
            device=self.args.device,
            )
        delete_list = sim_calc.cal_delete_list(sim_matrix, threshold=0.9, exclude_digonal=False)
        repeat_rate = len(delete_list)/len(sim_matrix)
        print(f"重复率大于0.9的图片数量：{len(delete_list)}，比例：{len(delete_list)/len(sim_matrix)}")
        return repeat_rate

    def run(self):

        enpo_dir = self.split_13000_into_train_val_test()


        self.split_73000_into_train_val_test(enpo_dir)

        # self.calculate_mix73000_self_repeat_rate()
  
        pass


@dataclass
class Args:
    device: str = "cuda:1"  # cuda, cuda:0, cuda:1 or cpu

if __name__ == '__main__':
    args = hai.parse_args(Args)
    mps = MixedPolypSplitor(args=args)
    mps.run()
    pass