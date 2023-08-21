# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:54:34 2021

@author: wanglei
"""

# AP算法
# 在Blob数据集上的实验

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
from sklearn import metrics


# 1. 获得数据集
#X, y = make_blobs(n_samples=n_samples,centers=4, random_state=123456)
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.50, random_state=0)
#绘制原始数据
plt.scatter(X[:,0], X[:,1], alpha=0.7, edgecolors='b')

#计算相似度矩阵
simi = []
for m in X:
    ##每个数据与所有数据的相似度列表，即矩阵中的一行
    temp = []
    for n in X:
         ##采用负的欧式距离计算相似度
        s =-np.sqrt((m[0]-n[0])**2 + (m[1]-n[1])**2)
        temp.append(s)
    simi.append(temp)


p = -50 ##
#p = np.median(simi)  ##15个中心


#2. 执行AP算法：
# 参数： 学习速率damping=0.5, 最大迭代=500， 
ap = AffinityPropagation(preference=p).fit(X)


#输出结果： 
labels = ap.labels_
cluster_centers_indices = ap.cluster_centers_indices_

#聚类中心的数量   
n_clusters_ = len(cluster_centers_indices)
print('聚类的数量: %d' % n_clusters_)


# 3. 性能评价
print("轮廓系数: %0.3f"% metrics.silhouette_score(X, y))


# 4 绘图

##根据聚类中心划分数据
c_list = []

for m in X:
    temp = []
    for j in cluster_centers_indices:
        n = X[j]
        d = -np.sqrt((m[0]-n[0])**2 + (m[1]-n[1])**2)
        temp.append(d)
    
    ##按照是第几个数字作为聚类中心进行分类标识
    c = cluster_centers_indices[temp.index(np.max(temp))]
    c_list.append(c)




colors = ['red','blue','black','green','yellow','cyan','magenta','red','blue','black','green','yellow','cyan','magenta',]
plt.figure(figsize=(8,6))
#plt.xlim([-3,3])
#plt.ylim([-3,3])

for i in range(X.shape[0]):
    d1 = X[i]
    d2 = X[c_list[i]]

    c = cluster_centers_indices.tolist().index(c_list[i])
    plt.plot([d2[0],d1[0]],[d2[1],d1[1]],color=colors[c],linewidth=1)
    
    if i == c_list[i]:
       plt.scatter(d1[0],d1[1],color=colors[c],linewidth=3)
    else:
       plt.scatter(d1[0],d1[1],color=colors[c],linewidth=1)

plt.show()
