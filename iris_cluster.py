# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 00:24:23 2022

@author: bbill
"""

from sklearn import datasets,metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
iris = datasets.load_iris()
X = iris.data
X = pd.DataFrame(X)
Nan = X.isnull().any()
print(Nan)
# 分群結果
tStart_kmeans = time.time()
kmeans= KMeans(n_clusters=3)
ow = kmeans.fit(X)
pre_kmeans = ow.predict(X)
tEnd_kmeans = time.time() 
pre_kmeans = np.array(pre_kmeans).reshape(-1,1)
# 品種
y_true = iris.target
y_true = np.array(y_true).reshape(-1,1)
# silhouette_avg = metrics.silhouette_score(y_true, pre_kmeans) #kmeans
# print(silhouette_avg)
def purity(cluster, label):   #純度
    cluster = np.array(cluster)
    label = np. array(label)
    indedata1 = {}
    for p in np.unique(label):
        indedata1[p] = np.argwhere(label == p)
    indedata2 = {}
    for q in np.unique(cluster):
        indedata2[q] = np.argwhere(cluster == q)
    count_all = []
    for i in indedata1.values():
        count = []
        for j in indedata2.values():
            a = np.intersect1d(i, j).shape[0]
            count.append(a)
        count_all.append(count)
    return sum(np.max(count_all, axis=0))/len(cluster)

# print("Purity：",purity_kmeans)

from sklearn.cluster import AgglomerativeClustering
agg=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='single')
# affinity: 距離的計算方式，”euclidean”,”l1″,”l2″,”manhattan”,”cosine”…
# linkage: 群與群之間的距離，”ward”,”complete”,”average”,”single”
tStart_hierach = time.time()
agg.fit(X)
pre_agg = agg.fit_predict(X)
tEnd_hierach = time.time()
pre_agg = np.array(pre_agg).reshape(-1,1)
from scipy.cluster.hierarchy import dendrogram, linkage
# Performs hierarchical/agglomerative clustering on X by using "Ward's method"
linkage_matrix = linkage(X, 'single')
figure = plt.figure(figsize=(7.5, 5))
# Plots the dendrogram
dendrogram(linkage_matrix, labels = pre_agg)
plt.title('Hierarchical Clustering Dendrogram (single)')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.tight_layout()
plt.show()
# silhouette_avg = metrics.silhouette_score(y_true, pre_agg)  #hierarchical
# print(silhouette_avg)
from sklearn.cluster import DBSCAN
tStart_dbscan = time.time()
dbscan=DBSCAN()
dbscan.fit(X)
tEnd_dbscan = time.time()
pre_dbscan = dbscan.labels_
# pre_dbscan = np.array(pre_dbscan).reshape(-1,1)
# print(pre_dbscan)
purity_kmeans = purity(y_true,pre_kmeans)
purity_agg = purity(y_true,pre_agg)
purity_dbscan = purity(y_true,pre_dbscan)

kmeanstimes = tEnd_kmeans - tStart_kmeans
hierachtimes = tEnd_hierach - tStart_hierach
dbscantimes = tEnd_dbscan - tStart_dbscan
df_c = pd.DataFrame({  'kmeans_purity':[purity_kmeans],'kmeans_time':[kmeanstimes]
                          ,'hierach_purity':[purity_agg],'hierach_time':[hierachtimes]
                         , 'dbscan_purity':[purity_dbscan],'dbscan_time':[dbscantimes]
                          }) 
print(df_c)

