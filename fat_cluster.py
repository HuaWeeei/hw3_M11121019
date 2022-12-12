# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 00:24:23 2022

@author: bbill
"""

from sklearn import datasets,metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
data=pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
X =data.iloc[:,0:18]
X = X.drop("family_history_with_overweight",axis=1)
Nan = X.isnull().any()
print(Nan)
y = data['family_history_with_overweight']
y = pd.DataFrame(y)
labelencoder = LabelEncoder()  #類別化
X['Gender']= labelencoder.fit_transform(X['Gender'])
X['FAVC']= labelencoder.fit_transform(X['FAVC'])
X['CAEC']= labelencoder.fit_transform(X['CAEC'])
X['SMOKE']= labelencoder.fit_transform(X['SMOKE'])
X['SCC']= labelencoder.fit_transform(X['SCC'])
X['CALC']= labelencoder.fit_transform(X['CALC'])
X['MTRANS']= labelencoder.fit_transform(X['MTRANS'])
X['NObeyesdad']= labelencoder.fit_transform(X['NObeyesdad'])
y['family_history_with_overweight']= labelencoder.fit_transform(y['family_history_with_overweight'])

#PCA降維至兩個維度(欄位)
pca = PCA(n_components=0.95) #百分之95%資料
X = pca.fit_transform(X)

# kmean
K_max= 50
scores=[]
ii = []
for i in range(2,K_max+1):
    kss=silhouette_score(X,KMeans(n_clusters=i).fit_predict(X))  #尋找較佳k值
    scores.append(kss)
    ii.append(i)
#得出最佳k值
selected_K = scores.index(max(scores)) + 2 
print('K=',selected_K)
plt.plot(ii, scores , lw=0.8, color="red", label="score")
plt.legend()
plt.title('Better_K')
plt.show()
tStart_kmeans = time.time()
kmeans= KMeans(n_clusters=selected_K)  #最佳k=2
ow = kmeans.fit(X)
pre_kmeans = ow.predict(X)
tEnd_kmeans = time.time() 
pre_kmeans = np.array(pre_kmeans).reshape(-1,1)
# 品種
y_true = y
y_true = np.array(y_true).reshape(-1,1)

# silhouette_avg = metrics.silhouette_score(y_true, pre_kmeans) #kmeans
# print(silhouette_avg)


# print("Purity：",purity_kmeans)
#hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
agg=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='average')
# affinity: 距離的計算方式，”euclidean”,”l1″,”l2″,”manhattan”,”cosine”…
# linkage: 群與群之間的距離，”ward”,”complete”,”average”,”single”
tStart_hierach = time.time()
agg.fit(X)
pre_agg = agg.fit_predict(X)
tEnd_hierach = time.time()
pre_agg = np.array(pre_agg).reshape(-1,1)
from scipy.cluster.hierarchy import dendrogram, linkage
# Performs hierarchical/agglomerative clustering on X by using "Ward's method"
linkage_matrix = linkage(X, 'average')   #使用average link 並畫圖
figure = plt.figure(figsize=(7.5, 5))
# Plots the dendrogram
dendrogram(linkage_matrix, labels = pre_agg)
plt.title('Hierarchical Clustering Dendrogram (Average)')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.tight_layout()
plt.show()
# silhouette_avg = metrics.silhouette_score(y_true, pre_agg)  #hierarchical
# print(silhouette_avg)
from sklearn.cluster import DBSCAN
tStart_dbscan = time.time()
res=[]
for eps in np.linspace(1,10,10,dtype=int):    #找尋dbscan較佳參數組合
    for min_samples in range(2,10):
        dbscan= DBSCAN(eps=eps,min_samples=min_samples)
        dbscan.fit(X)
        n_clusters = len([i for i in set(dbscan.labels_) if i !=-1])
        outliners = np.sum(np.where(dbscan.labels_ == -1,1,0))
        stats = str(pd.Series([i for i in dbscan.labels_ if i !=-1]).value_counts().values)
        res.append({'eps':eps,'min_samples':min_samples,'n_clusters':n_clusters,
                    'outliners':outliners,'stats':stats})
df = pd.DataFrame(res)
pre = df.loc[df.n_clusters ==2,:]      #指定只有兩組cluster
dbscan=DBSCAN(eps=5,min_samples=3)
tStart_ada= time.time()
dbscan.fit(X)
tEnd_dbscan = time.time()
pre_dbscan = dbscan.labels_
# pre_dbscan = np.array(pre_dbscan).reshape(-1,1)
# print(pre_dbscan)
#績效
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
purity_kmeans = purity(y_true,pre_kmeans)
purity_agg = purity(y_true,pre_agg)
purity_dbscan = purity(y_true,pre_dbscan)
acc_kmeans = (y_true == pre_kmeans).mean()
acc_agg = (y_true == pre_agg).mean()
acc_dbscan = (y_true == pre_dbscan).mean()
kmeanstimes = tEnd_kmeans - tStart_kmeans
hierachtimes = tEnd_hierach - tStart_hierach
dbscantimes = tEnd_dbscan - tStart_dbscan
df_c = pd.DataFrame({  'kmeans_purity':[purity_kmeans],'kmean_acc':[acc_kmeans]
                     ,'kmeans_time':[kmeanstimes]
                          ,'hierach_purity':[purity_agg],'hierach_acc':[acc_agg]
                          ,'hierach_time':[hierachtimes],'dbscan_purity':[purity_dbscan],
                          'dbscan_acc':[acc_dbscan],'dbscan_time':[dbscantimes]
                          }) 
print(df_c)

