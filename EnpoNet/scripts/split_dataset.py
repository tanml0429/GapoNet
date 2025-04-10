import os
import shutil
import random
import sys
import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class SplitDataset:
    def __init__(self):
        pass


    def hsv_convert(self, img_path):
        image = cv2.imread(img_path)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
        # 计算直方图
        hist_size = 256  # 选择直方图的bin数量
        hist_range = [0, 256]  # 设定直方图的取值范围
        # 分别计算H、S和V通道的直方图
        hist_h = cv2.calcHist([image_hsv], [0], None, [hist_size], hist_range)
        hist_s = cv2.calcHist([image_hsv], [1], None, [hist_size], hist_range)
        hist_v = cv2.calcHist([image_hsv], [2], None, [hist_size], hist_range)

        # 归一化直方图
        hist_h_normalized = cv2.normalize(hist_h, hist_h)
        hist_s_normalized = cv2.normalize(hist_s, hist_s)
        hist_v_normalized = cv2.normalize(hist_v, hist_v)

        # 将三个直方图连接成一个特征向量
        hist_feature = np.concatenate((hist_h_normalized, hist_s_normalized, hist_v_normalized), axis=None)

        return hist_feature     

    def split_dataset(self, hist_features, image_paths, target_path):
        # 数据预处理

        scaler = StandardScaler() # 标准化
        scaled_features = scaler.fit_transform(hist_features)  # 标准化特征
        # 训练K-means模型

        k = 2
        kmeans = KMeans(n_clusters=k)  # 聚类数目
        kmeans.fit(scaled_features)  # 训练模型


        # 预测每个特征的聚类标签
        labels = kmeans.predict(scaled_features)

        # 创建子文件夹来存储普通内镜图片和染色内镜图片
        normal_path = f'{target_path}/normal_endoscopy/images'
        n_labels_path = f'{target_path}/normal_endoscopy/labels'
        dyed_path = f'{target_path}/dyed_endoscopy/images'
        d_labels_path = f'{target_path}/dyed_endoscopy/labels'
        os.makedirs(f'{normal_path}', exist_ok=True)
        os.makedirs(f'{n_labels_path}', exist_ok=True)
        os.makedirs(f'{dyed_path}', exist_ok=True)
        os.makedirs(f'{d_labels_path}', exist_ok=True)

        #清空文件夹
        for path in [normal_path, n_labels_path, dyed_path, d_labels_path]:
            for root, dirs, files in os.walk(path):
                for file in files:
                    os.remove(os.path.join(root, file))
        
        
        # 将图片移动到对应的子文件夹中
        for i, label in enumerate(labels):
            image_path = image_paths[i]
            img_path = Path(image_path)

            # 图片对应的标签地址
            # 把imges文件夹替换成labels文件夹, 把.jpg替换成.txt
            label_path = f'{img_path}'.replace('/images/', '/labels/').replace('.jpg', '.txt')

            # label_path = f'{img_path.parent.parent}/label/{img_path.stem}.txt'
            file_name = os.path.basename(img_path)  
            if label == 0:
                shutil.copy(img_path, os.path.join(f'{dyed_path}', file_name))
                shutil.copy(label_path, os.path.join(f'{d_labels_path}', file_name.replace('.jpg', '.txt')))
            else:
                shutil.copy(img_path, os.path.join(f'{normal_path}', file_name))
                shutil.copy(label_path, os.path.join(f'{n_labels_path}', file_name.replace('.jpg', '.txt')))

        print (f'Normal endoscopy images: {len(os.listdir(normal_path))}')
        print (f'Dyed endoscopy images: {len(os.listdir(dyed_path))}')


if __name__ == "__main__":
    # 获取图片路径
    source_path = f'/data/tml/hybrid_polyp_v5_format'
    target_path = f'/data/tml/split_polyp_v5_format'
    # 获取所有图片路径
    image_paths = [str(path) for path in Path(source_path).glob("**/*.jpg")]  # 获取所有图片路径
    print (f'Found {len(image_paths)} images')
    hist_features = []
    for img_path in tqdm(image_paths):  # tqdm用于显示进度条
        data = SplitDataset()
        hist_feature = data.hsv_convert(img_path)
        hist_features.append(hist_feature)
    hist_features = np.array(hist_features)
    data.split_dataset(hist_features, image_paths, target_path)