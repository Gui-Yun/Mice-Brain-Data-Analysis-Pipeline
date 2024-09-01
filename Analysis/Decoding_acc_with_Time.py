import scipy.io
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file_folder = 'Data/M30_0420'

# 读取 trails.mat 文件
trails_data = scipy.io.loadmat(file_folder + '/Trail_data.mat')
trails = trails_data['trails']

# 读取 label.mat 文件
label_data = scipy.io.loadmat(file_folder + '/label.mat')
labels = label_data['labels'].flatten()

# 确定时间序列的长度和时间窗长度
time_series_length = 50
time_window_length = 5

# 计算最大时间窗数
max_windows = time_series_length // time_window_length

# 记录不同时间窗长度的准确率
accuracy_results = []

# 循环以不同时间窗长度训练模型
for num_windows in range(1, max_windows + 1):
    # 当前使用的时间点
    current_length = num_windows * time_window_length

    # 创建特征矩阵
    data = np.zeros((trails.shape[0], trails.shape[1] * current_length))

    for i in range(trails.shape[0]):
        for j in range(trails.shape[1]):
            # 这里取 trails[i, j] 中前 current_length 个数据点
            trail_segment = trails[i, j].flatten()
            trail_segment = trail_segment[-1-current_length:-1]
            data[i, j * current_length:(j + 1) * current_length] = trail_segment

    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

    # 定义并训练 SVM 模型
    model = svm.SVC(kernel='sigmoid')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 打印当前时间窗长度下的准确率
    print(f"Time window length: {current_length}, SVM Model accuracy: {accuracy * 100:.2f}%")

    # 记录准确率
    accuracy_results.append((current_length, accuracy * 100))

# 打印不同时间窗长度下的准确率结果
print("Accuracy results for different time window lengths:")
for length, acc in accuracy_results:
    print(f"Length {length}: {acc:.2f}%")
