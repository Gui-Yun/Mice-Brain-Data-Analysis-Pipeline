import scipy.io
import numpy as np
from sklearn import svm
import joblib


file_folder = 'Data/M30_0420'

# 加载训练好的SVM模型
model = joblib.load(file_folder + '/svm_model.joblib')

N = 1495
length = 50

support_vec = model.support_vectors_
decision_scores = []

for i in range(N):
    decision_vec = support_vec[(i-1)*length:i*length -1]
    decision_scores.append(np.linalg.norm(decision_vec))

    
# 保存所有样本的决策函数分数为MAT文件
output_filename = file_folder + '/decision_scores_model.mat'
scipy.io.savemat(output_filename, {'decision_scores': decision_scores})

print(f"Decision function scores have been saved to {output_filename}")