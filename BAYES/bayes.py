# coding=utf-8

from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1表示侮辱性文字, 0表示正常言论
    return postingList,classVec

def createWordList(dataset):
    WordList = set([])
    for words in dataset:
        WordList = WordList | set(words)
    return list(WordList)

def creatWordVector(WordList, inputSet):
    returnVector = [0]*len(WordList)
    for word in inputSet:
        if word in WordList:
            returnVector[WordList.index(word)] = 1 #index方法取出list里面对应的word的索引
        else:
            print "the word: %s is not in my vocabulary!" % word
    return returnVector

def train(List, Labels):
    numOfDocs = len(List) #用于训练的文档的数目
    numOfWords = len(List[0]) #单词表中单词的总数
    Pabusive = sum(Labels) / float(numOfDocs) #计算样本文档中侮辱性abusive的概率
    # P0num = zeros(numOfWords)
    # P1num = zeros(numOfWords)
    # P0Denom = 0.0
    # P1Denom = 0.0
    # 为避免出现0概率：
    P0num = ones(numOfWords)
    P1num = ones(numOfWords)
    P0Denom = 2.0
    P1Denom = 2.0
    for i in range(numOfDocs):
        if Labels[i] == 1:
            P1num += List[i] #循环累加计算了侮辱性文本中各个单词出现的次数
            P1Denom += sum(List[i]) #统计了侮辱性文本中单词总数
        else:
            P0num += List[i] #循环累加计算了正常文本中各个单词出现的次数
            P0Denom += sum(List[i]) #统计了正常文本中单词总数

    # P1vector = P1num / P1Denom #单词表中每个单词出现在侮辱性文本中的概率
    # P0vector = P0num / P0Denom #单词表中每个单词出现在正常文本中的概率
    # 为避免概率太小，最后乘积约等于0：
    # 机器学习实战一书中取log没有取相反数，导致概率变为负数，最后结果与预计相反！
    P0vector = -log(P1num / P1Denom)
    P1vector = -log(P0num / P0Denom)
    return P0vector ,P1vector ,Pabusive

def classify(DocVec, P0Vec, P1Vec, Pclass): #要分类的文本，单词分别是侮辱性或者正常的概率，样本是侮辱性的概率
    p1 = sum(DocVec * P1Vec) + (-log(Pclass)) #log相加，为相乘
    p0 = sum(DocVec * P0Vec) + (-log(1.0 - Pclass)) #1减去侮辱性文本的概率等于样本中正常文本的概率
    if p1 > p0:
        return 1
    else:
        return 0

