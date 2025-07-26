# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, make_blobs, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# -------------------- 向量相似度计算 --------------------

# 定义两个向量
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

# 余弦相似度
cos_sim = cosine_similarity([vector1], [vector2])
print(f"Cosine Similarity: {cos_sim[0][0]}")

# 欧氏距离
euclidean_dist = np.linalg.norm(vector1 - vector2)
print(f"Euclidean Distance: {euclidean_dist}")

# 曼哈顿距离
manhattan_dist = np.sum(np.abs(vector1 - vector2))
print(f"Manhattan Distance: {manhattan_dist}")

print("\n" + "-"*50 + "\n")

# -------------------- K近邻（KNN）分类 --------------------

# 加载数据集
data = load_iris()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化KNN模型（k=3）
knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy * 100:.2f}%")

print("\n" + "-"*50 + "\n")

# -------------------- K-means 聚类 --------------------

# 生成一个随机数据集
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# 使用K-means聚类，k=4
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# 获取聚类结果
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')  # 聚类中心点
plt.title("K-means Clustering")
plt.show()

print("\n" + "-"*50 + "\n")

# -------------------- 线性回归 --------------------

# 生成回归数据集
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练线性回归模型
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 预测
y_pred = regressor.predict(X_test)

# 绘制结果
plt.scatter(X_test, y_test, color='blue', label='True values')
plt.plot(X_test, y_pred, color='red', label='Predicted values')
plt.title("Linear Regression")
plt.legend()
plt.show()

# 输出模型的系数和截距
print(f"Coefficient: {regressor.coef_[0]}")
print(f"Intercept: {regressor.intercept_}")

print("\n" + "-"*50 + "\n")

# -------------------- 支持向量机（SVM）分类 --------------------

# 加载数据集
data = load_iris()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化SVM模型
svm = SVC(kernel='linear')

# 训练模型
svm.fit(X_train, y_train)

# 预测
y_pred = svm.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy * 100:.2f}%")

print("\n" + "-"*50 + "\n")

# -------------------- 决策树分类 --------------------

# 加载数据集
data = load_iris()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化决策树模型
dt = DecisionTreeClassifier(random_state=42)

# 训练模型
dt.fit(X_train, y_train)

# 预测
y_pred = dt.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")

print("\n" + "-"*50 + "\n")
