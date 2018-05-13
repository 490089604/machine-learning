#!/usr/bin/env python3
#上面是为了在linux和MAC环境下以exe形式运行
#coding=utf-8
#输出中文，不写会报错
# @Time    : 2018/5/9 14:51
# @Author  : Changfa Wu
# @Site    :
# @File    : knn.py
# @Software: PyCharm
#自己写的knn算法

from numpy import *
import operator

##给出训练数据以及对应的类别
def createDataSet():
    group = array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5]])#数组长度为0
    labels = ['A','A','B','B']
    return group,labels

###通过KNN进行分类。每个属性都计算欧式距离
def classify(input,dataSet,label,k):
    dataSize = dataSet.shape[0]#输出a的shape会显示一个参数，就是这个list中元素个数,array创建的可以看成一个list
    ####计算欧式距离
    diff = tile(input,(dataSize,1)) - dataSet#循环4次后相减
    print (diff)
    sqdiff = diff ** 2
    """
    没有axis参数表示全部相加，axis＝0表示按列相加，axis＝1表示按照行的方向相加
    """
    squareDist = sum(sqdiff,axis = 1)###行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5
    print(dist)
    ##对距离进行排序
    sortedDistIndex = argsort(dist)##argsort()根据元素的值从大到小对元素进行排序，返回下标
    print(sortedDistIndex)
    classCount={}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        ###对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    ###选取出现的类别次数最多的类别
    maxCount = 0
    for key,value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes
dataSet,labels =createDataSet()
input = array([1.1,0.3])
K = 3
output = classify(input,dataSet,labels,K)
print(output)
"""
测试样例
#-*-coding:utf-8 -*-
import sys
sys.path.append("...文件路径...")
import KNN
from numpy import *
dataSet,labels = KNN.createDataSet()
input = array([1.1,0.3])
K = 3
output = KNN.classify(input,dataSet,labels,K)
print("测试数据为:",input,"分类结果为：",output)
"""