import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

here = Path(__file__).parent

# 读取数据
npy_path = "/data/tml/enpo_dataset/similar_matrix/newdata_matrix.npy"
#判断是否包含filtered
if 'filtered' in npy_path:
    dataset = npy_path.split('/')[-1].split('_')[0] + '_filtered'
else:
    dataset = npy_path.split('/')[-1].split('_')[0]
data = np.load(npy_path)  # 假设数据保存在名为'your_data_file.npy'的文件中

# 对数据进行PCA降维
# n_components = 2
# batch_size = 1000
# ipca = IncrementalPCA(n_components=n_components)

# with tqdm(total=len(data)) as pbar:
#     for i in range(0, len(data), batch_size):
#         batch_data = np.load(npy_path, mmap_mode='r')[i:i+batch_size]
#         ipca.partial_fit(batch_data)
#         pbar.update(batch_size)

# data_pca = ipca.transform(data)
plt.figure(figsize=(10, 8))
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# 可视化降维后的数据
# plt.figure(figsize=(10, 8))
# with tqdm(total=len(data_pca)) as pbar:
#     for i in range(0, len(data_pca), batch_size):
#         batch_data_pca = data_pca[i:i+batch_size]
#         plt.scatter(batch_data_pca[:, 0], batch_data_pca[:, 1], alpha=0.5)
#         pbar.update(len(batch_data_pca))
plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'PCA Visualization of {dataset} Dataset')
# plt.show()
# 保存图片
plt.savefig(f'{here}/pca/{dataset}.png')

