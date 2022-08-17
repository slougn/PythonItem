#!/usr/bin/env python
# coding=utf-8
'''
Author: 560130
Date: 2022-06-07 16:18:01
LastEditTime: 2022-06-22 18:03:17
LastEditors: 560130
Description: 方法1,使用PCA+KNN
FilePath: /PythonItem/ML/Tang/Li1.py
'''
from mnist import MNIST
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# 导入数据
print("正在导入数据....")
mndata = MNIST(
    '/home/gree/wfDocument/gree/wfDocument/其他/练习项目/PythonItem/ML/Tang/mnist数据')
print("导入数据完毕！")
print("数据预处理...")
train_x, train_y = mndata.load_training()
# or
test_x, test_y = mndata.load_testing()

train_x = np.array(train_x)
test_x = np.array(test_x)
print("数据处理完毕！")
# 训练PCA
print("开始训练模型...")
pca = PCA(n_components=0.90)  # 保存95%的能量
new_train_x = pca.fit_transform(train_x)
print("训练完成！")
new_test_x = pca.transform(test_x)


# 识别、测试
print("使用分类器识别")

knn = KNeighborsClassifier(n_neighbors=3)  #定义一个knn分类器对象
knn.fit(new_train_x, train_y)  #调用对象的训练方法
test_y_predict = knn.predict(new_test_x)  #调用测试方法
acc = sum(test_y_predict == np.array(test_y)) / len(test_y)

#score = knn.score(new_test_x, test_y,sample_weight=None)

print("准确率为：",acc)
