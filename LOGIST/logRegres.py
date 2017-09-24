# coding=utf-8
from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('LOGIST/testSet.txt')
    for line in fr.readlines():
        oneline = line.strip().split() #读取每一行，去除空格，转化为列表
        dataMat.append([1.0, float(oneline[0]), float(oneline[1])]) #加入x0维度取值1
        labelMat.append(int(oneline[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1 + exp(-inX))

def graidentAscent(dataMat, classLabels):           #梯度上升法
    dataMatrix = mat(dataMat)   #mat()转化为numpy矩阵
    labelMatrix = mat(classLabels).transpose() #转置矩阵
    m, n = shape(dataMatrix) #获取矩阵长宽
    alpha = 0.001  #learning rate
    maxCycles = 500  #循环次数
    weights = ones((n,1)) #用1初始化权值
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)   #simoid归一化
        error = (labelMatrix - h)  #计算误差
        weights = weights + alpha * dataMatrix.transpose() * error  #更新权值
    return weights

def plotBestFit(weights):   #画图
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArray = array(dataMat)
    n = shape(dataArray)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArray[i,1]); ycord1.append(dataArray[i,2])
        else:
            xcord2.append(dataArray[i,1]); ycord2.append(dataArray[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def graidentAscent0(dataMat, classLabels):           #梯度上升法（改进为随机）
    # dataMatrix = mat(dataMat)   #mat()转化为numpy矩阵
    # labelMatrix = mat(classLabels).transpose() #转置矩阵
    m, n = shape(dataMat) #获取矩阵长宽
    alpha = 0.01  #learning rate
    # maxCycles = 500  #循环次数
    weights = ones(n) #用1初始化权值
    for k in range(m):
        h = sigmoid(sum(dataMat[k] * weights))   #simoid归一化
        error = classLabels[k] - h  #计算误差   h,error 算出来都是向量
        weights = weights + alpha * dataMat[k] * error  #更新权值
    return weights

def gradientAscent1(dataMat, classLabels, numIter=150):    #第二次优化
    # dataMatrix = mat(dataMat)   #mat()转化为numpy矩阵
    # labelMatrix = mat(classLabels).transpose() #转置矩阵
    m, n = shape(dataMat) #获取矩阵长宽
    # alpha = 0.01  #learning rate
    # maxCycles = 500  #循环次数
    weights = ones(n) #用1初始化权值
    for k in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + k + i) + 0.01 #调整alpha的值
            randIndex = int(random.uniform(0, len(dataIndex))) #随机选取更新
            h = sigmoid(sum(dataMat[randIndex] * weights))   #simoid归一化
            error = classLabels[randIndex] - h  #计算误差   h,error 算出来都是向量
            weights = weights + alpha * dataMat[randIndex] * error  #更新权值
            del(dataIndex[randIndex])  #避免重取
    return weights