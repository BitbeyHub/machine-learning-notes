# 随机森林



- [返回顶层目录](../../SUMMARY.md#目录)
- [返回上层目录](ensemble-learning.md)
- [随机森林概述](#随机森林概述)




[决策树与随机森林](http://www.cnblogs.com/fionacai/p/5894142.html)

[集成学习算法总结----Boosting和Bagging](http://lib.csdn.net/article/machinelearning/35135)

[Bagging与随机森林算法原理小结](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247487987&idx=2&sn=9389a64487c53fd59f09dd6b541ee9bc&chksm=ebb42927dcc3a03154332989ae6e07ce84af69b45f8120f8852f6111c655898472a752664fd3&mpshare=1&scene=1&srcid=0711bsXANeXJTAHrLxFmlBIn#rd)

随机森林就是bagging，用bagging的方法生成很多cart树，就是随机森林。

集成学习在机器学习算法中具有较高的准去率，不足之处就是模型的训练过程可能比较复杂，效率不是很高。目前接触较多的集成学习主要有2种：基于Boosting的和基于Bagging，前者的代表算法有Adaboost、GBDT、XGBOOST、后者的代表算法主要是随机森林。

# 随机森林概述

随机森林（Random Forest，RF）在以决策树为基学习器的基础上，进一步在决策树的训练过程中引入了随机属性选择。具体来说，传统决策树在选择划分属性时是在当前结点的属性集合（假定有d个属性）中选择一个最优属性；而在RF中，对基决策树的每个结点，先从该结点的属性集合中随机选择一个包含k个属性的子集，然后再从这个子集中选择一个最优属性用于划分，这里的参数k控制了随机性的引入程度。 

随机森林简单、容易实现、计算开销小，令人惊奇的是，它在很多现实任务中展现出强大的性能，被誉为“代表集成学习技术水平的方法”。

作者：Liangjun_Feng 
来源：CSDN 
原文：https://blog.csdn.net/liangjun_feng/article/details/78123583?utm_source=copy 
版权声明：本文为博主原创文章，转载请附上博文链接！



机器学习教程 之 随机森林： 算法及其特征选择原理

https://blog.csdn.net/liangjun_feng/article/details/80152796