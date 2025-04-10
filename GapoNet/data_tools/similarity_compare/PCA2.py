import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

# 获取当前文件的路径
here = Path(__file__).parent

# 准备两个数据集的路径
npy_path_1 = "/data/tml/enpo_dataset/similar_matrix/newdata_matrix.npy"
npy_path_2 = "/data/tml/enpo_dataset/similar_matrix/enpo_12927_origin.npy"

# 判断是否包含'filtered'并设置数据集名称
def get_dataset_name(npy_path):
    if 'filtered' in npy_path:
        return npy_path.split('/')[-1].split('_')[0] + '_filtered'
    return npy_path.split('/')[-1].split('_')[0]

dataset_1 = get_dataset_name(npy_path_1)
dataset_2 = get_dataset_name(npy_path_2)

# 读取数据
data_1 = np.load(npy_path_1)
data_2 = np.load(npy_path_2)

# 创建PCA对象进行降维
pca = PCA(n_components=2)

# 对第一个数据集进行PCA
data_pca_1 = pca.fit_transform(data_1)

# 对第二个数据集进行PCA
data_pca_2 = pca.fit_transform(data_2)

# 绘图
plt.figure(figsize=(10, 8))
plt.scatter(data_pca_2[:, 0], data_pca_2[:, 1], alpha=0.5, color='#1f77b4', label=f'{dataset_2} PCA')
plt.scatter(data_pca_1[:, 0], data_pca_1[:, 1], alpha=0.5, color='purple', label=f'{dataset_1} PCA')


# 添加图例和标签
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization of Two Datasets')
plt.legend()

# 保存图像
plt.savefig(f'{here}/results/{dataset_1}_{dataset_2}.png')

