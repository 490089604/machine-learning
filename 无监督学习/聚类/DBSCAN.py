#!/usr/bin/env python3
#上面是为了在linux和MAC环境下以exe形式运行
#coding=utf-8
#输出中文，不写会报错
# @Author  : Changfa Wu
import  numpy as np
from sklearn.cluster import  DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

def loadData(filePath):
    # fr = open(filePath, 'r+')
    with open(filePath, 'r+', encoding='utf-8') as fr:
        lines = fr.readlines()
    data = []  # 数据
    key = []  # 标志
    for line in lines:
        items = line.strip().split(",")  # 去掉前后空格和逗号
        key.append(items[0])
        data.append([float(items[i]) for i in range(1, len(items))])
    """
    https://blog.csdn.net/csj664103736/article/details/72828584
    X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据。
    X[:,  m:n]，即取所有数据的第m到n-1列数据，含左不含右
    X[n,:]是取第1维中下标为n的元素的所有值。
    X[1,:]即取第一维中下标为1的元素的所有值，输出结果：
    eg:X=real_X[1:3,0:1] 取2，3行的第一个数字
    """

    return key,data
if __name__ == '__main__' :
    #city.txt 仅为测试，无意义
    """
    eps: 两个样本被看作邻居节点的最大距离
	min_samples: 簇的样本数,决定核心节点
	metric：距离计算方式
    fit(X[,y,sample_weight])：训练模型。
    fit_predict(X[,y,sample_weight])：训练模型并预测每个样本所属的簇标记。
    """
    key,X = loadData('city.txt')
    db = DBSCAN(eps=5, min_samples=2).fit(X)#
    labels = db.labels_
    print('Labels:')#每个数据的簇标签
    print(labels)
    raito = len(labels[labels[:] == -1]) / len(labels)
    print('Noise raito:', format(raito, '.2%'))

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))#评价聚类效果。可以选择其他的。
    for i in range(n_clusters_):
        print('Cluster ', i, ':')
        temp_k=[labels == i][0]
        temp_list=[]
        for j in range(len(key)):
            if temp_k[j] == True:
                temp_list.append(key[j])
        print(temp_list)
    #plt.hist(X, 24)