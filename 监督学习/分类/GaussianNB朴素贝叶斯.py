#!/usr/bin/env python3
#coding=utf-8
# @Time    : 2018/5/9 18:32
# @Author  : Changfa Wu
# @Site    : 
# @File    : GaussianNB朴素贝叶斯.py
# @Software: PyCharm
import numpy as np
from sklearn.naive_bayes import GaussianNB
#朴素贝叶斯一般在小规模数据上的表现很好，适合进行多分类任务。
x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1,1,1,2,2,2])
#根据大数定律，当训练街包括充足的独立同分布样本时，p(c)可通过各类样本出现的频率进行估计
gnb = GaussianNB(priors=None)#priors ：给定各个类别的先验概率。如果为空，则按训练数据的实际情况进行统计；如果给定先验概率，则在训练过程中不能更改。
gnb.fit(x,y)

x= [[-2,-4],[1,5]]#测试时可以构造二维数组达到同时预测多个样本的目的
print(gnb.predict(x))