# XGBoost

* [返回顶层目录](../../SUMMARY.md#目录)
* [返回上层目录](ensemble-learning.md)
* [XGBoost概述](#XGBoost概述)


Xgboost只是陈天奇写的一个工具包，本身并不是一个算法。

传统GBDT以CART作为基分类器，xgboost还支持线性分类器，这个时候xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）；xgboost是Gradient Boosting的一种高效系统实现，并不是一种单一算法。

# XGBoost概述

最近引起关注的一个Gradient Boosting算法：XGBoost，在计算速度和准确率上，较GBDT有明显的提升。XGBoost 的全称是eXtreme Gradient Boosting，它是Gradient Boosting Machine的一个c++实现，作者为正在华盛顿大学研究机器学习的大牛陈天奇 。XGBoost最大的特点在于，它能够自动利用CPU的多线程进行并行，同时在算法上加以改进提高了精度。它的处女秀是Kaggle的 希格斯子信号识别竞赛，因为出众的效率与较高的预测准确度在比赛论坛中引起了参赛选手的广泛关注。值得我们在GBDT的基础上对其进一步探索学习。





