# coding=utf-8
# #2017.8.14

from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],
                   [1.0,1.0],
                   [0.0,0.0],
                   [0.0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):   #knn算法
    dataSetSize = dataSet.shape[0]  #行数，即样本总数
    newDataSet = tile(inX, (dataSetSize,1))  #创建行数个待测样本的集合
    diffMat = newDataSet - dataSet   #计算新样本-每个样本
    sqDiffMat = diffMat**2   #矩阵乘法，每一项计算平方
    sqDistances = sqDiffMat.sum(axis=1)  #矩阵加法，axis=1：每一行相加
    distances = sqDistances**0.5   #开平方的距离
    sortedID = distances.argsort()    #按升序排序序号
    classCount={}   #字典
    for i in range(k):   #取前k个最小值
        targetlabel = labels[sortedID[i]]
        classCount[targetlabel] = classCount.get(targetlabel,0) + 1    #get获取键为tragetlabel的值，没有则为0，
                                                                       #是在统计出现次数，没出现过初始化0
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    #按出现次数的降序排列，key=operator.itemgetter(1)指按字典值排序
    return sortedClassCount[0][0]   #返回最佳值

def file2matrix(filename):   #读文件转化为Numpy
    file = open(filename)
    lines = file.readlines()    #读入每一行的数据
    linenumber = len(lines)     #得到总行数
    returnMatrix = zeros((linenumber,3))   #创建要返回的数据矩阵，初始化为0
    labelsVector = []   #创建要返回的标签矩阵
    index = 0
    for line in lines:
        line = line.strip()   #截去所有回车字符
        listFromLine = line.split('\t')    #按tab键把数据切片分割
        returnMatrix[index,:] = listFromLine[0:3]    #每一行读入前三个数据到返回的数据矩阵中
        labelsVector.append(int(listFromLine[-1]))   #每一行最后一个读入返回的标签矩阵中
        index += 1
    return returnMatrix,labelsVector

def autoNormal(dataset):   #归一化特征值
    minValue = dataset.min(0)   #获取列的最小值
    maxValue = dataset.max(0)   #获取列的最大值
    range = maxValue - minValue
    normalDataset = zeros(shape(dataset))
    m = dataset.shape[0]
    normalDataset = dataset - tile(minValue,(m,1))
    normalDataset = normalDataset/tile(range,(m,1))   #不是矩阵除法
    return normalDataset,range,minValue

def datingClassTest():   #测试函数
    precent = 0.1    #取10%数据测试
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normalMatrix, ranges, minValue = autoNormal(datingDataMat)
    m = normalMatrix.shape[0]   #行数
    TestNumber = int(m*precent)    #取前10%的数据进行预测
    error = 0.0
    for i in range(TestNumber):
        predictResult = classify0(normalMatrix[i,:],normalMatrix[TestNumber:m,:],datingLabels[TestNumber:m],3)
        print "the classifier came back with:%d,the real answer is %d"%(predictResult, datingLabels[i])
        if(predictResult != datingLabels[i]):
            error += 1.0
    print "the total error rate is : %f" % (error/float(TestNumber))


def img2vector(filename):
    returnVect = zeros((1,1024))   #一行1024列0
    fr = open(filename)
    for i in range(32):   #依次读32行数据
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
