import scipy.io
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

# patched_Neron_data = scipy.io.loadmat('Output/patched_Nerons.mat')
# patched_Neron_ind = patched_Neron_data['indices']

# 在 1 到 1495 之间随机抽取 20 个不重复的索引
# random_indices = np.random.choice(range(1, 1496), size=20, replace=False)

# trails = trails[:,Neron_ind[0]]
# trails = trails[:,random_indices]

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


# 打乱数据和标签
# permuted_indices = np.random.permutation(len(labels))
# data = data[permuted_indices]
# labels = labels[permuted_indices]

# 打印labels以确保正确加载
print(f"labels: {labels}")

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

# 定义模型字典
models = {
    'SVM': svm.SVC(kernel='sigmoid')# ,
    # 'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    # 'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42),
    # 'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
    # 'Naive Bayes': GaussianNB()
}

# 循环训练和评估模型
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Model accuracy: {accuracy * 100:.2f}%")

# 保存模型到文件
joblib.dump(model, file_folder + '/svm_model.joblib')
print("模型已保存到 svm_model.joblib")