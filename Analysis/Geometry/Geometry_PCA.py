import scipy.io
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import mat73
from mpl_toolkits.mplot3d import Axes3D

file_folder = 'Data/M30_0420'

# 读取 trails.mat 文件
trails_data = mat73.loadmat(file_folder + '/wholebrain_output.mat')
data = trails_data['whole_trace_ori']

print(data.shape)

Neron_data = scipy.io.loadmat(file_folder + '/Nerons.mat')
Neron_ind = Neron_data['reliable_neurons']

print(Neron_ind)

data = data[Neron_ind[0],:]


# 1. 数据标准化
data_normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# 2. PCA降维到3维
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data_normalized.T)  # 转置数据，降维每个时间点的信号

# 3. 绘制神经流形
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 使用时间点作为颜色映射
time_points = np.arange(data_pca.shape[0])
sc = ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=time_points, cmap='viridis')

# 添加颜色条
plt.colorbar(sc, ax=ax, label='Time Points')

# 设置标签
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.title('Neural Manifold in 3D')
plt.show()

# 3. 查看每个主成分解释的方差比例
explained_variance_ratio = pca.explained_variance_ratio_
# 输出结果
print(f"每个主成分的方差解释比例: {explained_variance_ratio}")