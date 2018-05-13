#!/usr/bin/env python3
#上面是为了在linux和MAC环境下以exe形式运行
#coding=utf-8
#输出中文，不写会报错
# @Author  : Changfa Wu
import  numpy as np
from sklearn.cluster import  KMeans
import matplotlib.pyplot as plt
"""
NumPy是Python语言的一个扩充程序库。支持高级大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
"""
def loadData(filePath):
    #fr = open(filePath, 'r+')
    with open (filePath, 'r+',encoding='utf-8') as fr:
        lines = fr.readlines()
    retData = []#数据
    retCityName = []#城市名字
    for line in lines:
        items = line.strip().split(",")#去掉前后空格和逗号
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1, len(items))])
    """
    float(items[i]) for i in range(1, len(items))
    等价于
    for i in range(1,len(items)):
        pass
    """
    return retData,retCityName
"""
    开发文档http://scikit-learn.org/stable/modules/classes.html
    可能需要的参数
    n_clusters：用于指定聚类中心的个数
	init：初始聚类中心的初始化方法
	max_iter：最大的迭代次数
	一般调用时只用给出n_clusters即可，init默认是k-means++，max_iter默认是300
"""
if __name__ == '__main__' :
    #city.txt 仅为测试，无意义
    data, cityName = loadData('city.txt')
    km = KMeans(n_clusters=3)#聚类结果为三
    label = km.fit_predict(data)# 计算簇中心以及为簇分配序号，获得标签，0,1,2
    expenses = np.sum(km.cluster_centers_, axis=1) # axis按行求和，求得是聚类中心的支出
    CityCluster = [[], [], []]#定义三个list
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):#输出聚类结果，将结果输出到对应的距离中
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])
    """
    改进：
    源代码中用的欧几里得距离，并且没办法设置距离参数，所以需要更改源码。
    scipy.spatial.distance.cdist
    使用scipy.spatial.distance.cdist(A, B, metric=‘cosine’)、
    A：A向量
	B：B向量
	metric: 计算A和B距离的方法，更改此参数可以更改调用的计算距离的方法
	目前源码还没看明白，继续学习。
    """
