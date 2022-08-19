#!/usr/bin/env python
# coding=utf-8
'''
Author: 560130
Date: 2022-05-09 21:26:53
LastEditTime: 2022-05-13 00:46:47
LastEditors: 560130
Description: 
FilePath: /pythonitem/hgy/test/test.py
'''

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    '''
    description:sigmoid函数 
    param {x} x为np.array格式
    return {} 输出np.array格式
    '''
    sig = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    sig = sig + 1
    sig = sig * 0.5
    return sig

def smoothstep(x,edge0=-0.2,edge1=0.2):
    # if(x < edge0):
    #     return 0
    # if(x > edge1):
    #     return 1
    x[x<edge0] = 0
    x[(x>=edge0) & (x<=edge1)] = (x[(x>=edge0) & (x<=edge1)]-edge0)/(edge1-edge0)
    x[(x>=edge0) & (x<=edge1)] = x[(x>=edge0) & (x<=edge1)] * x[(x>=edge0) & (x<=edge1)] * (3 - 2 * x[(x>=edge0) & (x<=edge1)])
    x[x>edge1] = 1
    return x


def less_than(x,standard,y):
    '''
    description:小于某数,则变为y  if(x<standard) x=y
                使用sigmoid函数,以0为界限。
    param {*}  x为np.array格式,is input
               y是变为的数字
               standard 是小于的数
    return {*} y
    '''
    # 把sigmoid函数
    out = (-sigmoid(x-standard)+1)*y
    return out

def biger_than(x,standard,y):
    '''
    description: if(x>standard) x=y
    param {*}
    return {*}
    '''
    out = sigmoid(x-standard) * y
    return out

def bigger_less_than(x,standard1,standard2,y):
    '''
    description: if(standar1<x<standar2) x=y
    param {*}
    return {*}
    '''
    out = biger_than(x,standard1,y) * less_than(x,standard2,y)
    return out

if __name__ == "__main__":
    x = np.linspace(-10, 10, 2000)
    t = np.linspace(-10,10,2000)
    y = t.copy()
    # y[(t>=0.1)&(t<0.3)] = bigger_less_than(y[(t>=0.1)&(t<0.3)],0.1,0.3,0.3) 
    # y[t<0.1] = less_than(y[t<0.1],0.1,5)
    y[t>1] = biger_than(y[t>1],1,1)
    tmp = smoothstep(t)
    plt.plot(x,tmp)
    plt.show()



    
    #t[t<0.1] = 0
    #t[(t>=0.1) & (t<0.3)] = 0.3
    #t[t>1] = 1