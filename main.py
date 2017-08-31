# coding=utf-8

from numpy import *
import matplotlib
import  matplotlib.pyplot as plt


if __name__ == "__main__":
    # import KNN.knn
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
    # plt.legend()
    # plt.show()

    # normalMat, ranges, minValue = KNN.knn.autoNormal(datingDataMat)
    # print normalMat
    # print ranges
    # print minValue
    #
    # KNN.knn.datingClassTest()
    # KNN.knn.handwritingClassTest()
    ################################
    import TREES.trees
    data,labels = TREES.trees.createDataSet()
    print(data)
    print(TREES.trees.Xiangnong(data))
    print(TREES.trees.SelectDataset(data,0,1))
    # print(SelectDataset(data,0,0))
    # print(SelectDataset(data,2,'no'))
    # print(SelectDataset(data,2,'yes'))
    print(TREES.trees.ChooseBestFeature(data))
    print(TREES.trees.CreateTree(data,labels))

    import TREES.treesPlotter
    TREES.treesPlotter.createPlot()
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


