#!/usr/bin/env python
# coding=utf-8
'''
Author: 560130
Date: 2022-10-12 14:14:05
LastEditTime: 2022-10-12 14:35:22
LastEditors: 560130
Description: 《Dive into deep learning》中辅助画图的函数
FilePath: /PythonItem/ML/DiveIntoDeepLearning/d2lzh_pytorch.py
'''
from IPython import display
from matplotlib import pyplot as plt
import random
import torch


def use_svg_displty():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_displty()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


def squared_loss(y_hat, y):
    # 这里返回的是向量，另外，pytorch里的MSELoss没有除以2
    return (y_hat - y.view(y_hat.size()))**2 / 2


def sgd(params,lr,batch_size):
    for param in params:
        param.data -= lr * param.grad /batch_size # 这里更改param时用的param.data