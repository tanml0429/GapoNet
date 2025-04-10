import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
here = Path(__file__).parent
import os
import json
from tqdm import tqdm

# with open('/home/tml/VSProjects/polyp_mixed/polyp/data/hybrid_data/hybrid_data2.json', 'r') as f:
#     data = json.load(f)

# folder_path = '/data/tml/mixed_polyp_filtered/total_polyp_v5_format/images'

# hybrid_npy = np.load('/data/tml/similar_matrix/total_polyp_v5_format_filtered.npy')
# source_data = {}

# for i, img_name in enumerate(tqdm(os.listdir(folder_path), desc='Processing images')):
#     img_id = img_name.split('.')[0].split('_')[1]
#     found = False

#     for entry in data['entries']:
#         if img_id in entry['source']:
#             source = entry['source']
#             if source not in source_data:
#                 source_data[source] = []
#             source_data[source].append(hybrid_npy[i])
#             found = True
#             break
#     if not found:
#         print(f"Image {img_id} not found in data")


# for source, source_list in source_data.items():
#     source_data[source] = np.array(source_list) 
#     print(f"Source {source} has {source_data[source].shape[0]} images")
#     np.save(f'/data/tml/similar_matrix/total_filtered/{source}_filtered.npy', source_data[source])




# 读取数据
npy_paths = []
for npy_path in os.listdir('/data/tml/similar_matrix/total_filtered'):
    npy_paths.append(f'/data/tml/similar_matrix/total_filtered/{npy_path}')

colors = []
for i in range(len(npy_paths)):
    colors.append(plt.cm.get_cmap('tab20')(i))


plt.figure(figsize=(10, 8))

data_list = []

for i, npy_path in enumerate(npy_paths):
    if 'filtered' in npy_path:
        dataset = npy_path.split('/')[-1].split('_')[0] + '_filtered'
    else:
        dataset = npy_path.split('/')[-1].split('_')[0]
    data = np.load(npy_path)
    data_list.append(data)
    
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5, color=colors[i], label=f'{dataset}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization of Datasets')
plt.legend()
# plt.show()
# 保存图片
plt.savefig(f'{here}/pca/total_split.png')








