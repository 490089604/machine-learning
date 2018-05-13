#!/usr/bin/env python3
#coding=utf-8
# @Time    : 2018/5/9 16:57
# @Author  : Changfa Wu
# @Site    : 
# @File    : DecisionTree.py
# @Software: PyCharm
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score#交叉验证值函数
clf = DecisionTreeClassifier()#默认基尼系数
iris = load_iris()
result = cross_val_score(clf,iris.data,iris.target,cv=10)#10折交叉验证
print(result)
clf.fit(iris.data,iris.target)
x=[[6.7, 3.0,5.2, 2.3]]
print(clf.predict(x))