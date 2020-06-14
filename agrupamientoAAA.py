# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram

dataset = pd.read_excel('C:/Users/Ponce/Desktop/Capacidades.xlsx')
X = dataset.iloc[0:, 1:].values
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
#hc = model
model.fit(X)
labels = model.labels_

plt.scatter(X[labels==0, 0], X[labels==0, 1], s=50, marker='o', color='red')
plt.scatter(X[labels==1, 0], X[labels==1, 1], s=50, marker='o', color='blue')
plt.scatter(X[labels==2, 0], X[labels==2, 1], s=50, marker='o', color='green')
plt.scatter(X[labels==3, 0], X[labels==3, 1], s=50, marker='o', color='purple')
plt.scatter(X[labels==4, 0], X[labels==4, 1], s=50, marker='o', color='orange')
#y_hc = hc.fit_predict(X)
plt.show()