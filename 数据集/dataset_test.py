#coding:utf-8
from sklearn.datasets import load_boston
#机器学习加载数据集
#data, target = load_boston(return_X_y = True)#f返回数据和价格，return_x_y 是否返回价格
# print(data.shape)
# print(target.shape)
boston = load_boston()#返回数据，包括很多属性，数据和目标（标签吧）
print(boston.target.shape)

from sklearn.datasets import load_iris #鸢尾花数据集
iris = load_iris()
print(list(iris.target_names))
"""
手写数字数据集包括1797个0-9的手写数字数据，每个数字由8*8大小的矩阵构成，矩阵中值的范围是0-16，代表颜色的深度。
"""
from sklearn.datasets import load_digits #手写数字库
digits = load_digits(n_class= 5)#返回为0到4的数据
print(digits.data.shape)
#(1797, 64)
print(digits.target.shape)
#(1797, )
print(digits.images.shape)
#(1797, 8, 8)
import matplotlib.pyplot as plt
plt.matshow(digits.images[1])#画图，若限制了则循环
plt.show()