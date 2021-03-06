# coding=utf-8

import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

# def createPlot():
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111, frameon=False)
#     plotNode('决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
#     plotNode('叶节点',(0.8,0.1),(0.3,0.8),leafNode)
#     plt.show()

def getLeafNum(myTree): #获取节点数目，以确定x轴的长度
    LeafNum = 0
    firstLabel = myTree.keys()[0] #从第一个键开始
    secondDirection = myTree[firstLabel] #取出字典
    for key in secondDirection.keys(): #循环取出字典中的键
        if type(secondDirection[key]).__name__=='dict': #如果取出的键对应的值的类型为字典
            LeafNum += getLeafNum(secondDirection[key]) #则递归继续统计节点
        else:
            LeafNum += 1 #不是字典则表明到某一分支末节点
    return LeafNum

def getTreeDepth(myTree):
    TreeDepth = 0
    firstLabel = myTree.keys()[0] #从第一个键开始
    secondDirection = myTree[firstLabel] #取出字典
    for key in secondDirection.keys(): #循环取出字典中的键
        if type(secondDirection[key]).__name__ == 'dict':  # 如果取出的键对应的值的类型为字典
            thisTreeDepth = 1 + getTreeDepth(secondDirection[key]) #不断递归到最深节点处
        else:
            thisTreeDepth = 1 #不是字典则表明到某一分支末节点,更新此处深度为1
        if thisTreeDepth > TreeDepth: #获取最深的节点深度
            TreeDepth = thisTreeDepth
    return TreeDepth

def plotMidText(cntrPt, parentPt, txtString): #在父子节点之间填充文本信息
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    width = getLeafNum(myTree)
    depth = getTreeDepth(myTree)
    firstLabel = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(width))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstLabel, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstLabel]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getLeafNum(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    plotTree(inTree, (0.5,1.0), '')
    plt.show()


def retrieveTree(i):
    listOfTrees =[
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]