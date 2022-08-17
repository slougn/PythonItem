#!/usr/bin/env python
# coding=utf-8
'''
Author: 560130
Date: 2022-06-07 16:17:07
LastEditTime: 2022-06-16 16:46:12
LastEditors: 560130
Description: 使用全连接网络
FilePath: /PythonItem/ML/Tang/tang1.py
'''
from mnist import MNIST
import numpy as np
import scipy.special


# 神经网络类定义
class neuralNetwork:

    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes,
                 learningrate) -> None:
        # 设置输入层、隐含层、输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 设置学习率
        self.lr = learningrate

        # 连接权重矩阵
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5),
                                    (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5),
                                    (self.hnodes, self.inodes))

        self.activation_function = lambda x: scipy.special.expit(x)

    # 训练神经网络
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, input)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_inputs

        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot(
            (output_errors * final_outputs *
             (1.0 - final_outputs)), np.transpose(hidden_errors))
        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs *
             (1.0 - hidden_outputs)), np.transpose(inputs))

    # 查询神经网络
    def query(self, inputs_list):
        # convert inputs list to 2d array
        input = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, input)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == "__main__":
    # 导入数据
    mndata = MNIST(
        '/home/gree/wfDocument/gree/wfDocument/其他/练习项目/PythonItem/ML/Tang/mnist数据'
    )

    train_x, train_y = mndata.load_training()
    # or
    test_x, test_y = mndata.load_testing()

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    # 配置
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    # 学习率
    learning_rate = 0.3

    # 创建神经网络实例
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    print(n.query([1.0, 0.5, -1.5]))
