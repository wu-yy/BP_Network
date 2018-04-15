#-*- coding:utf-8 -*-
import re
import xlrd
import xdrlib,sys
import xlwt
import datetime
import  time

if __name__=="__main__":
    data = xlrd.open_workbook("watermelon3.0.xlsx")
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    type1 = table.row_values(0)
    number=len(type1)
    data={}
    yR={}
    for i in range(number):
        yR[i]=[]
        yR[i].append(table.row_values(nrows-1)[i]-1)
        if i not in data.keys():
            data[i]=[]
            for k in range(nrows-1):
                data[i].append(table.row_values(k)[i])
    print(data)  # 获取到的数据
    print(yR)

    import matplotlib.pyplot as plt

    X=[]
    y=[]
    for i in range(12):
        X.append(data[i])
        y.append(yR[i])
    print(X)
    print(y)
    '''
    BP implementation
    '''
    from BP_network import *
    import matplotlib.pyplot as plt

    nn = BP_network()  # build a BP network class
    nn.CreateNN(8, 8, 1, 'Sigmoid')  # build the network

    e = []
    for i in range(2000):
        err, err_k = nn.TrainStandard(X, y, lr=0.5)
        e.append(err)
    f2 = plt.figure(2)
    plt.xlabel("epochs")
    plt.ylabel("accumulated error")
    plt.title("circles convergence curve")
    plt.plot(e)
    plt.show()

    '''
    draw decision boundary
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    XP=[]
    for i in range(0,17):
        XP.append(data[i])

    z = nn.PredLabel(XP)
    print(z)

