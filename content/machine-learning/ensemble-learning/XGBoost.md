# XGBoost

* [返回顶层目录](../../SUMMARY.md#目录)
* [返回上层目录](ensemble-learning.md)
* [XGBoost概述](#XGBoost概述)


首先**XGBoost**是Gradient Boosting的一种高效系统实现，只是陈天奇写的一个工具包，本身并不是一种单一算法。传统GBDT以CART作为基分类器，而XGBoost里面的基学习器除了用tree(gbtree)，也可用线性分类器(gblinear)，这个时候xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。

XGBoost可以看作是对GradientBoost的优化。其原理还是基于GradientBoost，它的创新之处是用了二阶导数和正则项。

xgboost是从决策树一步步发展而来的：

* 决策树 ⟶ 对样本重抽样，然后多个树平均 ⟶ Tree bagging
* Tree bagging ⟶ 再同时对特征进行随机挑选 ⟶ 随机森林
* 随机森林 ⟶ 对随机森林中的树进行加权平均，而非简单平均⟶ Boosing (Adaboost, GradientBoost)
* boosting ⟶ 对boosting中的树进行正则化 ⟶ xgboosting

从这条线一路发展，就能看出为什么xgboost的优势了。



这2个并不是独立的算法。

GBDT直接拿一阶导数来作为下一棵决策树的预测值，进行学习（具体树怎么学习不负责）；

xgboost则是拿一阶和二阶导数直接作为下一棵决策树的增益score指导树的学习



GBDT主要对loss L(y,F)关于F求梯度，利用回归树拟合该负梯度；

XGBOOST主要对loss L(y,F)二阶泰勒展开，然后求解析解，以解析解obj作为标准，贪心搜索split树是否obj更优。



补充个理解上很重要的点，之前的GBM模型（GBDT、GBRank、LambdaMART等）都把Loss加在的树间而未改动单棵CART内部逻辑（或者说无伤大雅懒得改），XGBoost因为正则化要考虑优化树复杂度的原因，把Loss带入到CART分裂目标和节点权重上去了（或者说把树内和树间的优化目标统一了）。作者：崔岩链接：https://www.zhihu.com/question/41354392/answer/397044000来源：知乎著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



\#模型：

gbdt仅支持CART（分类/回归树）ensemble(即addtive learning)模型

xgboost还支持线性/逻辑回归模型。

\#目标函数

这是最大的不同。

1. gbdt使用了 square loss的一阶求导，并且没有树本身因素的正则化部分。
2. xg使用了square loss的一阶和二阶求导，使用了树叶子节点（T）和叶子权重（w平方）作为正则化。**这里是xg最精髓的部分，它将基于样本的loss转化为了基于叶子节点的loss，即完成了参数的转变，这样才能 将loss部分和正则部分都转为叶子节点T的目标方程**。

\#优化算法：

大概一致，

1. 多棵树之间都是addtive learning，下颗树的目标是拟合 之前树的求和 和 目标间的差距。
2. 一颗树内的生成树方法也大概一致。都是基于最小化目标函数，去最大化每一次信息增益从而确定分裂点和分裂点权重。 

\#其他细节：

全是xg的。

1. 正则化相关：shrinkage等
2. 目标函数的可定制化：比如支持样本权重。
3. NULL值的特殊处理，作为一个特殊值来处理（实践中很有用）





gradient版本将f类比于参数，通过f对负梯度进行回归，通过负梯度逐渐最小化Object目标；xgboost版本通过使得当前Object目标最小化，构造出回归树f，更直接。两者都是求得f对历史累积F进行修正。



[一步一步理解GB、GBDT、xgboost](https://www.cnblogs.com/wxquare/p/5541414.html) 



XGBoost的原理http://djjowfy.com/2017/08/01/XGBoost%E7%9A%84%E5%8E%9F%E7%90%86/

[《xgboost导读和实战》](http://ishare.iask.sina.com.cn/f/h7zHjMF0z4.html#)

[维基百科 Gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting)

[GBDT算法原理与系统设计简介](http://wepon.me/files/gbdt.pdf)



[陈天奇论文《XGBoost: A Scalable Tree Boosting System》](https://arxiv.org/pdf/1603.02754v1.pdf)

[陈天奇论文演讲《Introduction to Boosted Trees 》](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)

[GBM之GBRT总结（陈天奇论文演讲翻译版）](http://nanjunxiao.github.io/2015/08/05/GBM%E4%B9%8BGBRT%E6%80%BB%E7%BB%93/)

[XGBoost官网](https://xgboost.readthedocs.io/en/latest/)





GBDT本质只过不就是函数空间上的梯度下降法，使用了损失函数的一阶导数。

# XGBoost概述

最近引起关注的一个Gradient Boosting算法：XGBoost，在计算速度和准确率上，较GBDT有明显的提升。XGBoost 的全称是eXtreme Gradient Boosting，它是Gradient Boosting Machine的一个c++实现，作者为正在华盛顿大学研究机器学习的大牛陈天奇 。XGBoost最大的特点在于，它能够自动利用CPU的多线程进行并行，同时在算法上加以改进提高了精度。它的处女秀是Kaggle的 希格斯子信号识别竞赛，因为出众的效率与较高的预测准确度在比赛论坛中引起了参赛选手的广泛关注。值得我们在GBDT的基础上对其进一步探索学习。



# 模型





# 损失函数

 

# 求解算法







# XGBoost比GradientBoost的区别及优势

为啥Xgboost比GradientBoost好那么多？

http://sofasofa.io/forum_main_post.php?postid=1000331