#!/usr/bin/env python
# coding=utf-8
'''
Author: 560130
Date: 2022-06-08 10:24:11
LastEditTime: 2022-06-22 18:03:15
LastEditors: 560130
Description: 使用LDA方法识别
FilePath: /PythonItem/ML/Tang/Li2.py
'''

from mnist import MNIST
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
print("数据处理完毕！")
# 训练LDA
print("开始训练模型...")
model = LinearDiscriminantAnalysis()
model.fit(train_x, train_y)
print("训练完成！")
# 测试
print("使用分类器识别")
test_y_predict = model.predict(test_x)
acc = sum(test_y == test_y_predict) / len(test_y)
print("准确率为：",acc)