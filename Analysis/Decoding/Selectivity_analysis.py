import numpy as np
from scipy.io import loadmat, savemat

# 文件夹
file_folder = 'Data/M30_0420'

import numpy as np
from scipy.io import loadmat

# 读取MAT文件
mat_data = loadmat(file_folder + '/Trail_data.mat')
trails = mat_data['trails']  # 假设这里存储了cell数组

# 读取刺激类型
label_data = loadmat(file_folder + '/label.mat')  # 扁平化数组
labels = label_data['labels']

labels = labels.flatten()

# 确保stimuli_types与responses_cell的长度匹配
if len(labels) != trails.shape[0]:
    raise ValueError("stimuli_types的长度与responses_cell的行数不匹配")

# 创建空的数组来存储神经元反应
mean_responses = np.zeros(trails.shape)


# 将cell数据转换为numpy数组
for i in range(trails.shape[0]):
   for j in range(trails.shape[1]):
    mean_responses[i][j] = np.mean(trails[i][j])

# 计算每个神经元对刺激类型1和2的平均反应
type_1_indices = np.where(labels == 1)[0]
type_2_indices = np.where(labels == 2)[0]

mean_responses_1 = np.mean(mean_responses[type_1_indices],axis=0)
mean_responses_2 = np.mean(mean_responses[type_2_indices],axis=0)

# 计算选择性指数
denominator = mean_responses_1 + mean_responses_2
selectivity_index = (mean_responses_1 - mean_responses_2) / denominator

# 输出选择性指数
print(selectivity_index)

# 保存选择性指数为MAT文件
output_filename = file_folder + '/selectivity_index.mat'
savemat(output_filename, {'selectivity_index': selectivity_index})

print(f"Selectivity index has been saved to {output_filename}")