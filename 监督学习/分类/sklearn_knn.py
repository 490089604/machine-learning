#!/usr/bin/env python3
#上面是为了在linux和MAC环境下以exe形式运行
#coding=utf-8
#输出中文，不写会报错
# @Time    : 2018/5/9 14:51
# @Author  : Changfa Wu
# @Site    : 
# @File    : sklearn_knn.py
# @Software: PyCharm
from sklearn.neighbors import KNeighborsClassifier#knn算法
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
knn =  KNeighborsClassifier(n_neighbors= 3)#调用最近的三个
knn.fit(X,y)#训练
data = [[1.1],[3.1]]
print(knn.predict(data))#预测
