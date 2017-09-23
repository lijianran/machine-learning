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

def createWordList(dataset): #创建单词列表
    WordList = set([])
    for words in dataset:
        WordList = WordList | set(words)
    return list(WordList)

def creatWordVector(WordList, inputSet): #创建文本词汇向量
    returnVector = [0]*len(WordList)
    for word in inputSet:
        if word in WordList:
            #returnVector[WordList.index(word)] = 1 #index方法取出list里面对应的word的索引
            returnVector[WordList.index(word)] += 1 #词袋模型
        else:
            print "the word: %s is not in my vocabulary!" % word
    return returnVector

def train(List, Labels): #训练算法
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

def textParse(bigString): #解析文本巨型字符串为单词列表
    import re
    Parse = re.compile('\\W*') #创建选择字母的正则表达式，以便进行选词，不选入符号
    listOfTokens = Parse.split(bigString) #进行分割，取词
    # listOfTokens = re.split(r'\W', bigString) #等同于上两行操作
    return [token.lower() for token in listOfTokens if len(token) > 2]
      #.lower()小写化字母，去除长度小于3的字母

def spamTest():
    docList = []; classList = []; fullText = [] #文章列表 类别列表 所有文章单词总和列表
    for i in range(1, 26):
        wordsList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordsList)
        fullText.extend(wordsList)
        classList.append(1)

        wordsList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordsList)
        fullText.extend(wordsList)
        classList.append(0)
    wordList = createWordList(docList) #文章列表拿去分析，去除重复单词，得到词汇表

    trainingSet = range(50); testSet = [] #训练集，测试集的索引值
    for i in range(10): #取出十个当作测试集
        randIndex = int(random.uniform(0,len(trainingSet))) #random.uniform(a,b)随机生成a到b之间的随机数
        testSet.append(trainingSet[randIndex]) #将随机取出的测试集序号加入到测试集中
        del(trainingSet[randIndex]) #在测试集中去除
    trainMatrix = []; trainClasses = [] #训练数据及标签的矩阵
    for i in trainingSet: #训练集序号
        trainMatrix.append(creatWordVector(wordList, docList[i]))  #取出相应序号的文章单词列表生成词汇向量
        trainClasses.append(classList[i]) #加入相应的文章的标签

    p0Vector, p1Vector, pSpam = train(trainMatrix, trainClasses) #算出算法需要的概率值
    errorcount = 0
    for i in testSet: #测试集序号
        wordVector = creatWordVector(wordList, docList[i])
        if classify(wordVector, p0Vector, p1Vector, pSpam) != classList[i]:
            errorcount += 1
    print "the error rate is: ",float(errorcount) / len(testSet)
