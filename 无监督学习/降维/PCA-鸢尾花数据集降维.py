#!/usr/bin/env python3
#上面是为了在linux和MAC环境下以exe形式运行
#coding=utf-8
#输出中文，不写会报错
# @Author  : Changfa Wu注释
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
 
data = load_iris()#以字典形式加载鸢尾花数据集
y = data.target#用y表示标签
X = data.data#用x表示数据
pca = PCA(n_components=2)#设置降维后为2维
reduced_X = pca.fit_transform(X)#对原始数据进行降维，保存在reduced_X中
 
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
print(reduced_X)
for i in range(len(reduced_X)):#将降维后的数据进行分类
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
 
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()