from yolov8 import YOLOv8  
import cv2
import os
import numpy as np
import shutil

# 加载YOLOv8模型
model = YOLOv8("path_to_yolov8_model")

# 设置相似度阈值
threshold = 0.9

# 遍历数据集中的所有图片
image_folder = 'path_to_image_folder'
image_files = os.listdir(image_folder)

# 创建用于存储删除的相似帧的文件夹
deleted_images_folder = 'path_to_deleted_images_folder'
os.makedirs(deleted_images_folder, exist_ok=True)

# 提取图像特征并存储成矩阵的形式
features_matrix = []
for image_file in image_files:
    image = cv2.imread(os.path.join(image_folder, image_file))
    features = model.extract_features(image)
    features_matrix.append(features)

features_matrix = np.array(features_matrix)

# 定义余弦相似度函数
def cosine_similarity_matrix(matrix):
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / norm
    similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)
    return similarity_matrix

# 计算相似度矩阵
similarity_matrix = cosine_similarity_matrix(features_matrix)

# 标记要删除的图片
delete_flags = np.any(similarity_matrix > threshold, axis=1)

# 将没有被标记删除的图片复制到新文件夹中
for image_file, delete_flag in zip(image_files, delete_flags):
    if not delete_flag:
        shutil.copy(os.path.join(image_folder, image_file), os.path.join(deleted_images_folder, image_file))

print(f"Non-similar images moved to {deleted_images_folder}")

