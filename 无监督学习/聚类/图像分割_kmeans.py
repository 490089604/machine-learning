#!/usr/bin/env python3
#上面是为了在linux和MAC环境下以exe形式运行
#coding=utf-8
#输出中文，不写会报错
# @Author  : Changfa Wu
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans#如果要用DBSCAN的话，参数不好选取
def loadData(filePath):
    with open(filePath,'rb') as f:
        img = image.open(f)
        data = []
        m,n = img.size
        for i in range(m):
            for j in range(n):
                #x,y,z = img.getpixel((i,j))
                x, y,z = img.getpixel((i, j))#三通道的
                data.append([x/256.0,y/256.0,z/256.0])
                print(i,j)
    return np.mat(data),m,n #将一个list换成一个矩阵
 
imgData,row,col = loadData('ock.jpg')
label = KMeans(n_clusters=6).fit_predict(imgData)
label = label.reshape([row,col])
pic_new = image.new("L", (row, col))
#pic_new = image.new("RGB", (row, col))
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j), int(256/(label[i][j]+1)))#向ij位置放置灰度值
pic_new.save("result-ock.jpg", "JPEG")
print('完啦')
