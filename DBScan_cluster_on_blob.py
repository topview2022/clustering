# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 22:45:46 2020

@author: wanglei
"""

#DBSCAN聚类模型在模拟blob数据集上的聚类实验

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs   
import matplotlib.pyplot as plt

# 1. 获得数据集
n_samples = 750 #数量
X, y = make_blobs(n_samples=n_samples, centers=3,random_state=123456)

#2. DBSCAN模型创建和训练
#模型参数
#  eps: 邻居半径（默认0.5）
#  min_samples: minPts(成为核心样本时半径内的最少样本数）
db = DBSCAN( eps=0.8, metric= 'euclidean',min_samples=10)
db.fit(X)
# 聚类结果,-1代表样本点是噪声点,其余值表示样本点所属的类（0，1）
labels = db.labels_
# 聚类数量
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

#3. 聚类模型评价
print('Estimated number of clusters: %d' % n_clusters_)
#轮廓系数
print("Silhouette Coefficient: %0.3f"  % metrics.silhouette_score(X, labels))
#Rand指数
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(y, labels))


##############################################################################
# 4. 绘图
#4.1 获得核心对象的掩码
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

#4.2 绘制原始数据
plt.figure(figsize=(9, 9))
plt.scatter(X[:,0], X[:,1],marker='o',color= 'b')
plt.title("original data")

#4.3 绘制DBSCAN的聚类结果
plt.figure(figsize=(9, 9))
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
i = -1
marker = ['v','^','o','x']  #依次表示第0，1类，噪声的marker
for k, col in zip(unique_labels, colors):
    if k == -1:    
        col = 'k'  # 黑色表示标记噪声点.
    class_member_mask = (labels == k)

    i += 1
    if (i>=len(unique_labels)):    
        i = 0

    # 绘制核心对象
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], marker[i], markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    # 绘制非核心对象和噪声
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], marker[i], markerfacecolor=col,
             markeredgecolor='k', markersize=6)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
