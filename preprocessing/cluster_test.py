import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,adjusted_rand_score,confusion_matrix,classification_report

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=iris.feature_names)
df['Species'] = y
df['Species_names'] = df['Species'].map({0:'setosa', 1:'versicolor', 2:'virginica'})

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=2,n_init = 10)
cluster_labels = kmeans.fit_predict(X)

df['Cluster'] = cluster_labels

print(iris.DESCR)
print(X.shape) #150 * 4维的矩阵
for i in range(10):
    for j in range(X.shape[1]):
        print(X[i][j],end = ' ')
    print()
print(X[:,:2])


# #1 可视化1--绘制散点图，观察真实标签和聚类结果的散点图相似度
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
# print(X_pca)
# df['PCA1'] = X_pca[:,0]
# df['PCA2'] = X_pca[:,1]
#
# plt.figure(figsize=(12, 5))
#
# # 子图1：根据真实标签着色
# plt.subplot(1, 2, 1)
# scatter = plt.scatter(df['PCA1'], df['PCA2'], c=df['Species'], cmap='viridis')
# plt.xlabel('First Principal Component')
# plt.ylabel('Second Principal Component')
# plt.title('True Labels')
# plt.legend(handles=scatter.legend_elements()[0], labels=target_names.tolist())
#
# # 子图2：根据聚类标签着色
# plt.subplot(1, 2, 2)
# scatter = plt.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis')
# plt.xlabel('First Principal Component')
# plt.ylabel('Second Principal Component')
# plt.title('K-Means Clusters')
# plt.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i}' for i in range(n_clusters)])
#
# plt.tight_layout()
# plt.show()


# 可视化2 -- 绘制轮廓分析图
silhouette_avg = silhouette_score(X, cluster_labels)
print(f'轮廓系数（silhouette score = {silhouette_avg:.3f}）')
#轮廓系数[-1,1]，越高越好，衡量一个样本与自身簇的相似度 vs 与其他簇的相似度

from sklearn.metrics import silhouette_samples

sample_silhouette_values = silhouette_samples(X, cluster_labels)
print(sample_silhouette_values.shape)

# 绘制轮廓分析图: 每个“刀片”代表一个簇。刀片的宽度代表簇的大小，高度代表每个样本的轮廓系数值。理想情况下，所有刀片都应该高于平均值（虚线），且宽度大致相等。
plt.figure(figsize=(10, 6))
y_lower = 10
for i in range(n_clusters):
    # 获取簇i的所有样本的轮廓系数
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    print(ith_cluster_silhouette_values.shape)
    print(ith_cluster_silhouette_values)
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.viridis(float(i) / n_clusters)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # 标记每个簇的轮廓系数平均值
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    y_lower = y_upper + 10

plt.title("Silhouette plot for the various clusters")
plt.xlabel("Silhouette coefficient values")
plt.ylabel("Cluster label")
plt.axvline(x=silhouette_avg, color="red", linestyle="--")  # 平均线
plt.yticks([])  # 清除y轴刻度
plt.show()



# #可视化3 -- 绘制热力图
# # 调整兰德指数 (Adjusted Rand Index, ARI) - 越高越好，最大为1
# # 衡量两个数据分配之间的一致性，考虑偶然因素
# ari = adjusted_rand_score(y, cluster_labels)
# print(f"调整兰德指数 (ARI): {ari:.3f}")
#
# # 混淆矩阵 (Confusion Matrix) - 查看聚类与真实类别的对应关系
# cm = confusion_matrix(y, cluster_labels)
# print("Confusion Matrix:")
# print(cm)
#
# # 可以使用热图更好地可视化混淆矩阵
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=[f'Cluster {i}' for i in range(n_clusters)],
#             yticklabels=target_names)
# plt.title('Confusion Matrix: Clusters vs True Labels')
# plt.ylabel('True Label')
# plt.xlabel('Cluster Label')
# plt.show()

centroids = kmeans.cluster_centers_
centroid_df = pd.DataFrame(data=centroids, columns=iris.feature_names)
print('Cluster Centroids(特征平均值):')
print(centroid_df.round(2))



