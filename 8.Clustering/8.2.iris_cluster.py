# coding:utf-8

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

data = pd.read_csv('iris.data', header=None, names=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度', '类别'])
x = data[['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']]
model = KMeans(n_clusters=3, init='k-means++')
model.fit(x)
y_pred = model.predict(x)
print('homogeneity_score = ', homogeneity_score(data['类别'], y_pred))
print('completeness_score = ', completeness_score(data['类别'], y_pred))
print('v_measure_score = ', v_measure_score(data['类别'], y_pred))
data['Predict'] = y_pred
print(data)
data.to_csv('result.csv', sep=',', encoding='gbk', index=False)
print('Data Save OK....')
