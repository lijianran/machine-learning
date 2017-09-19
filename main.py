# coding=utf-8

if __name__ == "__main__":
    # import KNN.knn
    # from numpy import *
    # import matplotlib.pyplot as plt
    #
    # group, labels = KNN.knn.createDataSet()
    # print KNN.knn.classify0([0.2,1.5], group, labels, 3)
    #
    # datingDataMat, datingLabels = KNN.knn.file2matrix('KNN\datingTestSet2.txt')
    # print datingDataMat
    # print datingLabels[0:20]
    #
    # from pylab import mpl
    # mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    # mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)  #参数349的意思是：将画布分割成3行4列，图像画在从左到右从上到下的第9块
    # ax.set_title(u'散点图')
    # plt.xlabel(u'玩视频游戏所耗时间百分比')
    # plt.ylabel(u'每年获取的飞行常客里程数')
    # #plt.ylabel(u'每周消耗的冰淇淋公升数')
    # ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*array(datingLabels),15.0*array(datingLabels))
    #
    # plt.show()
    #
    # normalMat, ranges, minValue = KNN.knn.autoNormal(datingDataMat)
    # print normalMat
    # print ranges
    # print minValue
    #
    # KNN.knn.datingClassTest()
    #
    # KNN.knn.handwritingClassTest()
    ################################
    # import TREES.tree
    # data,labels = TREES.tree.createDataSet()
    # print(data)
    # print(TREES.tree.xiangnong(data))
    # print(TREES.tree.splitDataSet(data,0,1))
    #
    # print(TREES.tree.chooseBestFeatureToSplit(data))
    # myTree = TREES.tree.createTree(data,labels)
    # print(myTree)
    #
    # import TREES.treePlotter
    #
    # #TREES.treePlotter.createPlot()
    # print(TREES.treePlotter.getLeafNum(myTree))
    # print(TREES.treePlotter.getTreeDepth(myTree))
    #
    # TREES.treePlotter.createPlot(myTree)
    # import TREES.treePlotter
    # print(TREES.treePlotter.retrieveTree(1))
    # myTree = TREES.treePlotter.retrieveTree(0)
    # TREES.treePlotter.createPlot(myTree)
    # import TREES.tree
    # data, labels = TREES.tree.createDataSet()
    # print(TREES.tree.classify(myTree, labels, [1,0]))
    # print(TREES.tree.classify(myTree, labels, [1,1]))
    #
    # TREES.tree.packTree(myTree, 'tree.txt')
    # print(TREES.tree.unpackTree('TREES/tree.txt'))
    #
    # file = open('TREES/lenses.txt')
    # lenses = [inst.strip().split('\t') for inst in file.readlines()]
    # lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # lensesTree = TREES.tree.createTree(lenses, lensesLabels)
    # print lensesTree
    # TREES.treePlotter.createPlot(lensesTree)
    ##############################
    # matplotlib画图
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # X, Y, Z = axes3d.get_test_data(0.05)
    # ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
    # cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    # cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    # cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    #
    # ax.set_xlabel('X')
    # ax.set_xlim(-40, 40)
    # ax.set_ylabel('Y')
    # ax.set_ylim(-40, 40)
    # ax.set_zlabel('Z')
    # ax.set_zlim(-100, 100)
    #
    # plt.show()
    ################################
    # data,label = loadDataSet()
    # weights = gradAscent(data,label)
    # print(weights)
    ################################
    import BAYES.bayes
    list ,labels = BAYES.bayes.loadDataSet()
    wordlist = BAYES.bayes.createWordList(list)
    print wordlist

    # vector = BAYES.bayes.creatWordVector(wordlist, list[0])
    # print vector

    trainMat = []
    for postinDoc in list:
        trainMat.append(BAYES.bayes.creatWordVector(wordlist, postinDoc))
      #将所给的文档全部向量化放在一个list中
    p0Vector, p1Vector, pAbusive = BAYES.bayes.train(trainMat, labels)
      #分别计算出单词表中每个单词在侮辱性文本和正常文本中出现的概率

    test1 = ['love', 'my', 'dalmation']
    Doc1 = BAYES.bayes.creatWordVector(wordlist, test1)
    if BAYES.bayes.classify(Doc1, p0Vector, p1Vector, pAbusive) == 1:
        print test1, "classified as ABUSIVE!"
    else:
        print test1, "classified as NORMAL!"

    test2 = ['stupid', 'garbage']
    Doc2 = BAYES.bayes.creatWordVector(wordlist, test2)
    if BAYES.bayes.classify(Doc2, p0Vector, p1Vector, pAbusive) == 1:
        print test2, "classified as ABUSIVE!"
    else:
        print test2, "classified as NORMAL!"
