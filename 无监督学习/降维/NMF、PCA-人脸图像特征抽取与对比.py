#!/usr/bin/env python3
#上面是为了在linux和MAC环境下以exe形式运行
#coding=utf-8
#输出中文，不写会报错
# @Author  : Changfa Wu注释
from numpy.random import RandomState#加载Olivetti人脸数据集导入函数
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces#加载Olivetti人脸数据集导入函数
from sklearn import decomposition
 
 
n_row, n_col = 2, 3#设置图像战术使得排列情况
n_components = n_row * n_col
image_shape = (64, 64)
 
 
###############################################################################
# Load faces data
#If True the order of the dataset is shuffled to avoid having images of the same person grouped.
dataset = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0))
#0在这里是伪随机数产生器的种子，也就是“the starting point for a sequence of pseudorandom number”，伪随机数相同，产生的种子就相同
faces = dataset.data
 #加载数据并打乱顺序，防止一个人的图片聚在一起

###############################################################################
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row)) #创建图片，指定图片大小
    plt.suptitle(title, size=16)#设置标题以及自豪大小
 
    for i, comp in enumerate(images):#利用它可以同时获得索引和值
        plt.subplot(n_row, n_col, i + 1)#选择画的子图
        vmax = max(comp.max(), -comp.min())#
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest', vmin=-vmax, vmax=vmax)#对数值进行归一化，并以灰度值形式显示
        # plt.show()
        # plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
        #            interpolation='nearest')
        # plt.show()
        '''
        如果显示分辨率与图像分辨率不一致（最常见的情况），
        那么interpolation ='nearest'只是简单地显示图像，而不尝试在像素之间进行插值。 这将导致像素显示为多个像素的平方的图像。
        vmin=-vmax, vmax=vmax，Min-Max normalization应该是将数值进行标准化处理然后限定到这个区间里
        '''
        #plt.show()
        plt.xticks(())#去除x轴，y轴
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.94, 0.04, 0.)#对子图的位置间隔进行调整
 
     
plot_gallery("First centered Olivetti faces", faces[:n_components])#取前6个，
###############################################################################
 
estimators = [
    ('Eigenfaces - PCA using randomized SVD',
         decomposition.PCA(n_components=6,whiten=True)),#降低后的维数为6维
 
    ('Non-negative components - NMF',
         decomposition.NMF(n_components=6, init='nndsvda', tol=5e-3))
]
 
###############################################################################
 
for name, estimator in estimators:
    print("Extracting the top %d %s..." % (n_components, name))
    print(faces.shape)
    estimator.fit(faces)
    components_ = estimator.components_ ##返回模型的各个特征向量,提取特征，这个是返回的特征向量，相当于矩阵W,H中的W
    #reduced_X = estimator.fit_transform(faces)这个是每个数据降维后的数值
    plot_gallery(name, components_[:n_components])#这是将行数缩减进行特征提取。
 
plt.show()