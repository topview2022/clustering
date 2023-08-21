# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:05:33 2023

@author: Thinkpad
"""
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs    # 用于生成数据集的库
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from util import draw_ellipse, BIC         # 引用辅助函数

# 1. 获得数据集
n_samples = 200                        # 样本数量
X, y = make_blobs(n_samples = n_samples, random_state = 9, centers = 4, cluster_std = 1)

# 2. GMM模型的创建和训练
K = 4                                  # 簇的数量
model = GaussianMixture(n_components = K, covariance_type = 'full', random_state = 15)
y_pred = model.fit_predict(X)

# 3. 聚类模型评价
print(" 轮廓系数: %0.3f" % metrics.silhouette_score(X, y_pred))
print(" 调整兰德指数AMI: %0.3f" % metrics.adjusted_rand_score(y, y_pred))

# 4绘图显示GMM的聚类结果

set_marker=['s','o','D','v','>','p','h']
set_color=['m','b','g','r','c','k','tan']
plt.figure(figsize = (5, 5))
plt.rcParams['font.sans-serif'] = ['SimHei']             #显示中文标签
plt.rcParams['axes.unicode_minus'] = False

for i in range(K):
    plt.scatter(X[y_pred == i][:, 0], X[y_pred == i][:, 1],  marker=set_marker[i], color=set_color[i])
    # 为簇绘制椭圆阴影区域
    for p, c, w in zip(model.means_, model.covariances_, model.weights_):
        draw_ellipse(p, c, alpha = 0.05)

plt.title(" GMM的聚类结果, K=%d"% K, fontsize = 14)
plt.show()


