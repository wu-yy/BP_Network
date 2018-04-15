#-*- coding:utf-8 -*-

def Sigmoid(x):
    from math import exp
    return 1.0 / (1.0 + exp(-x))
def SigmoidDerivate(y): #导函数
    return y * (1 - y)
def Tanh(x):
    from math import tanh
    return tanh(x)
def TanhDerivate(y):
    return 1 - y * y

def rand(a, b):
    '''
    random value generation for parameter initialization
    @param a,b: the upper and lower limitation of the random value
    '''
    from random import random
    return (b - a) * random() + a
class BP_network:

    def __init__(self):
        #每一层节点的数目
        self.i_n=0
        self.h_n=0
        self.o_n=0

        #每一层的输出值向量
        self.i_v=[]
        self.h_v=[]
        self.o_v=[]

        #权重和阈值
        self.ih_w=[] #权重
        self.ho_w=[]
        self.h_t=[] #阈值
        self.o_t=[]

        #定义选择函数和误差函数
        self.fun={
            'Sigmoid':Sigmoid,
            'SigmoidDerivate':SigmoidDerivate,
            'Tanh':Tanh,
            'TanhDerivate':TanhDerivate
        }

    def CreateNN(self,ni,nh,no,actfun): #分别代表输入数目、隐藏数目、输出数目、激活函数

            import numpy as np
            self.i_n=ni
            self.h_n=nh
            self.o_n=no

            #初始化每一层的输出
            self.i_v=np.zeros(self.i_n)
            self.h_v=np.zeros(self.h_n)
            self.o_v=np.zeros(self.o_n)

            #初始化连接的权重（随机初始化）
            self.ih_w=np.zeros([self.i_n,self.h_n])
            self.ho_w=np.zeros([self.h_n,self.o_n])
            for i in range(self.i_n):
                for h in range(self.h_n):
                    self.ih_w[i][h]=rand(0,1)

            for h in range(self.h_n):
                for o in range(self.o_n):
                    self.ho_w[h][o]=rand(0,1)

            #随机初始化每一层的阈值
            self.h_t=np.zeros(self.h_n)
            self.o_t=np.zeros(self.o_n)
            for h in range(self.h_n):
                self.h_t[h]=rand(0,1)
            for o in range(self.o_n):
                self.o_t[o]=rand(0,1)

            #初始化激活函数
            self.af=self.fun[actfun]
            self.afd=self.fun[actfun+'Derivate']

    def Pred(self,x):
        #x 输入的向量
        for i in range(self.i_n):
            self.i_v[i]=x[i]
        #激活隐藏层
        for h in range(self.h_n):
            total=0.0
            for i in range(self.i_n):
                total+=self.i_v[i]*self.ih_w[i][h]
            self.h_v[h]=self.af(total-self.h_t[h])
        #激活输出层
        for  j in range(self.o_n):
            total=0.0
            for h in range(self.h_n):
                total+=self.h_v[h]*self.ho_w[h][j]
            self.o_v[j]=self.af(total-self.o_t[j])

    def BackPropagate(self,x,y,lr):
        #计算BP算法
        #x,y 输入、输出、lr 学习率
        import numpy as np
        #获取当前网络的输出
        self.Pred(x)
        #计算 gradient
        # calculate the gradient based on output
        o_grid = np.zeros(self.o_n)
        for j in range(self.o_n):
            o_grid[j] = (y[j] - self.o_v[j]) * self.afd(self.o_v[j])

        h_grid = np.zeros(self.h_n)
        for h in range(self.h_n):
            for j in range(self.o_n):
                h_grid[h] += self.ho_w[h][j] * o_grid[j]
            h_grid[h] = h_grid[h] * self.afd(self.h_v[h])

            # updating the parameter
        for h in range(self.h_n):
            for j in range(self.o_n):
                self.ho_w[h][j] += lr * o_grid[j] * self.h_v[h]

        for i in range(self.i_n):
            for h in range(self.h_n):
                self.ih_w[i][h] += lr * h_grid[h] * self.i_v[i]

        for j in range(self.o_n):
            self.o_t[j] -= lr * o_grid[j]

        for h in range(self.h_n):
            self.h_t[h] -= lr * h_grid[h]

    def TrainStandard(self, data_in, data_out, lr=0.05):
        '''
        standard BP training
        @param lr, learning rate, default 0.05
        @return: e, accumulated error
        @return: e_k, error array of each step
        '''
        e_k = []
        for k in range(len(data_in)):
            x = data_in[k]
            y = data_out[k]
            self.BackPropagate(x, y, lr)

            # error in train set for each step
            y_delta2 = 0.0
            for j in range(self.o_n):
                y_delta2 += (self.o_v[j] - y[j]) * (self.o_v[j] - y[j])
            e_k.append(y_delta2 / 2)

        # total error of training
        e = sum(e_k) / len(e_k)

        return e, e_k

    def PredLabel(self, X):
        '''
        predict process through the network

        @param X: the input sample set for input layer
        @return: y, array, output set (0,1 - class) based on [winner-takes-all]
        '''
        import numpy as np

        y = []
        min=2
        for m in range(len(X)):
            self.Pred(X[m])
            print(self.o_v)
            if self.o_v[0]<min:
                min=self.o_v[0]
            if self.o_v[0] > 0.5:
                y.append(2)
            else:
                y.append(1)
        # max_y = self.o_v[0]
        #             label = 0
        #             for j in range(1,self.o_n):
        #                 if max_y < self.o_v[j]: label = j
        #             y.append(label)
        return np.array(y)
