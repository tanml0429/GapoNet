
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取指定的相似度矩阵
# similarity_matrix = np.load('/data/tml/enpo_dataset/similar_matrix/total_73856_origin.npy')
# similarity_matrix = np.load('/data/tml/enpo_dataset/similar_matrix/enpo_12927_origin.npy')
similarity_matrix = np.load('/data/tml/enpo_dataset/similar_matrix/newdata_matrix.npy')
# 将对角线以下的数据设置为NaN,包括对角线
similarity_matrix[np.tril_indices(similarity_matrix.shape[0])] = np.nan

#similarity_matrix[np.tril_indices(similarity_matrix.shape[0], -1)] = np.nan

# 展平矩阵为一维数组
data = similarity_matrix[~np.isnan(similarity_matrix)]

data = data 
# 统计数据的最大值、最小值、均值、中位数、标准差
max_value = np.max(data)
min_value = np.min(data)
mean_value = np.mean(data)
median_value = np.median(data)
std_value = np.std(data)
print('Max:', max_value)
print('Min:', min_value)
print('Mean:', mean_value)
print('Median:', median_value)
print('Std:', std_value)



# 绘制小提琴图并保存
sns.violinplot(y=data, color='purple')
plt.xlabel('Data')
plt.ylabel('Similarity')
plt.title('Violin Plot of Similarity Matrix')
# plt.show()
plt.savefig('/home/tml/VSProjects/GapoNet/GapoNet/data_tools/violet_plot/newdata2.png')
