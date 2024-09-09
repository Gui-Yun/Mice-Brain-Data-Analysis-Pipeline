import scipy.io
import numpy as np
from sklearn import svm
import joblib

file_folder = 'Data/M30_0420'

# 读取trails.mat文件
trails_data = scipy.io.loadmat(file_folder + '/Trail_data.mat')
trails = trails_data['trails']

# 读取label.mat文件
label_data = scipy.io.loadmat(file_folder + '/label.mat')
labels = label_data['labels']

Neron_data = scipy.io.loadmat(file_folder + '/Nerons.mat')
Neron_ind = Neron_data['reliable_neurons']

# 打印trails和labels的形状以确保正确加载
print(f"trails shape: {trails.shape}")
print(f"labels shape: {labels.shape}")

# 获取最大cell中的长度，以便创建统一大小的数组
max_length = max(trails[i, j].size for i in range(trails.shape[0]) for j in range(trails.shape[1]))

# 将每个cell的内容展开并拼接成一个特征向量
data = np.zeros((trails.shape[0], trails.shape[1] * max_length))

for i in range(trails.shape[0]):
    for j in range(trails.shape[1]):
        start_index = j * max_length
        length = trails[i, j].size
        data[i, start_index:start_index + length] = trails[i, j].flatten()

# 假设labels是一个一维数组，直接转换为numpy array
labels = labels.flatten()

# 加载训练好的SVM模型
model = joblib.load(file_folder + '/svm_model.joblib')

# 计算所有样本的决策函数分数
decision_scores = model.decision_function(data)

# 打印一个样本的决策函数分数作为检查
print(f"Decision function scores for sample 150: {decision_scores[149]}")

# 保存所有样本的决策函数分数为MAT文件
output_filename = file_folder + '/decision_scores.mat'
scipy.io.savemat(output_filename, {'decision_scores': decision_scores})

print(f"Decision function scores have been saved to {output_filename}")
