import cv2
import os
import numpy as np
from sklearn.decomposition import PCA

# 加载数据集
image_folder = 'path_to_image_folder'
image_files = os.listdir(image_folder)

# 读取图像并转换为灰度图
images = [cv2.imread(os.path.join(image_folder, file), cv2.IMREAD_GRAYSCALE) for file in image_files]

# 将图像展平成一维数组
data = [img.flatten() for img in images]

# 使用PCA进行降维
pca = PCA(n_components=50)  # 选择降维后的特征数量
pca.fit(data)
transformed_data = pca.transform(data)

# 从降维后的数据重构原始数据
reconstructed_data = pca.inverse_transform(transformed_data)

# 计算重构误差
reconstruction_errors = np.mean((data - reconstructed_data) ** 2, axis=1)

# 设置重构误差阈值
threshold = 1000

# 删除相似帧
deleted_files = []
for i, error in enumerate(reconstruction_errors):
    if error < threshold:
        file_to_delete = image_files[i]
        os.remove(os.path.join(image_folder, file_to_delete))
        deleted_files.append(file_to_delete)

print(f"Deleted {len(deleted_files)} files due to similarity")
