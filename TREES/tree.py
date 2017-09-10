# coding=utf-8

from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def xiangnong(dataSet):
    size = len(dataSet) #整体数据的数目
    labelCounts = {} #统计各个标签的次数
    for i in dataSet:
        currentLabel = i[-1] #取出每一个特征
        if currentLabel not in labelCounts.keys(): #特征第一次统计先初始化为0
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1 #然后统计出现次数
    xiangnong = 0.0
    for key in labelCounts: #循环总和所有特征的香农熵
        prob = float(labelCounts[key]) / size #计算每个特征出现的频率
        xiangnong -= prob * log(prob, 2) #计算香农熵
    return xiangnong

def splitDataSet(dataSet, axis, value): #分离选中特征
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] #取出0到axis的数据
            reducedFeatVec.extend(featVec[axis + 1:]) #在后面追加axis+1到最后的数据
            retDataSet.append(reducedFeatVec) #返回分离了第axis列数据的数据集
    return retDataSet

def chooseBestFeatureToSplit(dataSet): #选取最好的用于分类的特征
    numFeatures = len(dataSet[0]) - 1 #可以用于分类的特征的总数目
    baseXiangnong = xiangnong(dataSet) #先计算数据集基础原始的香农熵
    k = 0.0
    bestFeature = -1
    for i in range(numFeatures): #计算每一个特征用于分类之后的香农熵的变化，选取当前数据集最佳的分离特征
        featureList = [example[i] for example in dataSet] #取出每一个特征的可能取值
        uniqueFeature = set(featureList)  #唯一化取值
        newXiangnong = 0.0
        for value in uniqueFeature: #计算每一个特征值用来分类后的香农熵
            subDataSet = splitDataSet(dataSet, i, value) #分离出某特征的某一个取值后的数据集
            prob = len(subDataSet) / float(len(dataSet))
            newXiangnong += prob * xiangnong(subDataSet)
        change = baseXiangnong - newXiangnong #计算变化值
        if (change > k):  #每次循环都更新为信息复杂度更低的即香农熵更小的特征
            k = change
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet] #取出当前数据集的所有特征的取值
    if classList.count(classList[0]) == len(classList): #如果第一个特征的这一取值总数已经等于统计的总数
        return classList[0]  #说明已经分类完成
    if len(dataSet[0]) == 1:  #如果数据集只剩下一列，说明使用完所有特征都不能把原有数据集分离出仅包含唯一类别的分组
        return majorityCnt(classList) #挑选出出现次数最多的类别作为返回值
    bestFeature = chooseBestFeatureToSplit(dataSet) #选出当前数据集最佳用于分类的特征
    bestFeatLabel = labels[bestFeature]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeature])
    featValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:] #复制原有标签，以免更改了原有标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree

def classify(inputTree, featureLabels, test):
    firstLabel = inputTree.keys()[0] #从第一个键开始
    secondDirectory = inputTree[firstLabel] #取出字典
    featureIndex = featureLabels.index(firstLabel) #index查找当前列表中第一个匹配Label的元素
    # key = test[featureIndex]
    # value = secondDirectory[key]
    # if isinstance(value,dict): #isinstance是Python中的一个内建函数，用来判断一个对象是否是一个已知的类型
    #     classLabel = classify(value, featureLabels, test)
    # else:
    #     classLabel = value
    for key in secondDirectory.keys():
        if test[featureIndex] == key:
            if type(secondDirectory[key]).__name__ == 'dict':
                classLabel = classify(secondDirectory[key], featureLabels, test)
            else:
                classLabel = secondDirectory[key]
    return classLabel

