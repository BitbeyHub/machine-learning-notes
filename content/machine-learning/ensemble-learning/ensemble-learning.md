# 集成学习



* [返回顶层目录](../../SUMMARY.md#目录)
* [集成学习概述](#集成学习概述)
* [Bagging](#Bagging)
  * [随机森林](random-forest[Bagging].md)
* [Boosting](#Boosting)
  * [AdaBoost](AdaBoost[Boosting].md)
  * [GradientBoosting](#GradientBoosting)
    * [GBDT](GBDT[GradientBoosting].md)
    * [XgBoost](XgBoost[GradientBoosting].md)
* [Stacking](#Stacking)




[集成学习（ensemble learning ）该如何入门？](https://www.zhihu.com/question/29036379)



[《机器学习》笔记-集成学习（8）](https://mp.weixin.qq.com/s?__biz=MzUyMjE2MTE0Mw==&mid=2247485821&idx=1&sn=9cb901cb9c5144a1714eed4927c4b609&chksm=f9d157e5cea6def32faea122cf4193a77cfb3397543e95392acd6ec0185a647a42d2eab01ccd&mpshare=1&scene=1&srcid=032606p3UBT1losMDZ9DyjdX#rd)



[如何评价周志华深度森林模型](https://zhuanlan.zhihu.com/p/36621482)







# 集成学习概述

俗话说：三个臭皮匠，顶个诸葛亮。集成学习就是几个臭皮匠集成起来比诸葛亮还牛逼。







# Bagging

Bagging就是“Bootstrap aggregating（有放回采样的集成）”的缩写。Bagging是集成学习的一种，可提高用于统计分类回归的机器学习算法的稳定性和精度，也可以减小方差，有助于防止过拟合。虽然Bagging常使用决策树（即随机森林），但是也可用于任何方法，如朴素贝叶斯等。Bagging是模型平均方法（model averaging approach）的特例。

Bagging能防止噪声影响，是因为它的样本是有放回采样，这样子一些特例就很可能不会被采集到，即使采集到，也因为投票而被否决。这样就从样本上防止了过拟合。

Bagging样本的权值是一样的，各分类器的权值也都相等（即一人一票）。

# Boosting

数据有权重，分类器也有权重。给数据分权重了，分错的话，权重会增加，该数据就越重要；同样也给臭皮匠（分类器）分等级了，有不同的地位，分类器准确率越高，权值就越大。

弱分类器一般是naïve biyes和决策树。

## Boosting概述

沈学华 ,周志华 ,吴建鑫 ,等. Boosting 和 Bagging 综述

提升的含义就是**将容易找到的识别率不高的弱分类算法提升为识别率很高的强分类算法**，

Boosting方法是一种用来提高弱分类算法准确度的方法，这种方法通过构造一个预测函数系列,然后以一定的方式将他们组合成一个预测函数。Boosting是一种提高任意给定学习算法准确度的方法。它的思想起源于 Valiant提出的 [PAC](https://baike.baidu.com/item/PAC) ( Probably Approxi mately Correct)学习模型。

## Boosting算法起源

Boosting是一种提高任意给定学习算法准确度的方法。它的思想起源于 Valiant提出的 PAC ( Probably Approximately Correct)学习模型。Valiant和 Kearns提出了弱学习和强学习的概念，识别错误率小于1/2，也即准确率仅比随机猜测略高的学习算法称为弱学习算法；识别准确率很高并能在[多项式时间](https://baike.baidu.com/item/%E5%A4%9A%E9%A1%B9%E5%BC%8F%E6%97%B6%E9%97%B4)内完成的学习算法称为强学习算法。同时，Valiant和 Kearns首次提出了 PAC学习模型中弱学习算法和强学习算法的等价性问题，即任意给定仅比随机猜测略好的弱学习算法，是否可以将其提升为强学习算法？如果二者等价，那么只需找到一个比随机猜测略好的弱学习算法就可以将其提升为强学习算法，而不必寻找很难获得的强学习算法。1990年，Schapire最先构造出一种多项式级的算法，对该问题做了肯定的证明，这就是最初的Boosting算法。一年后，Freund提出了一种效率更高的Boosting算法。但是，这两种算法存在共同的实践上的缺陷，那就是都要求事先知道弱学习算法学习正确的下限。1995年，Freund和schapire改进了Boosting算法，提出了AdaBoost (Adaptive Boosting)算法，该算法效率和Freund于1991年提出的Boosting算法几乎相同，但不需要任何关于弱学习器的先验知识，因而更容易应用到实际问题当中。之后，Freund和schapire进一步提出了改变 Boosting投票权重的 AdaBoost . M1、AdaBoost . M2等算法，在机器学习领域受到了极大的关注。

## Boosting方法概述

Boosting方法是一种用来提高弱分类算法准确度的方法，这种方法通过构造一个预测函数系列，然后以一定的方式将他们组合成一个预测函数。它是一种框架算法，主要是通过对样本集的操作获得样本子集，然后用弱分类算法在样本子集上训练生成一系列的基分类器。他可以用来提高其他弱分类算法的识别率，也就是将其他的弱分类算法作为基分类算法放于Boosting框架中，通过Boosting框架对训练样本集的操作，得到不同的训练样本子集，用该样本子集去训练生成基分类器；每得到一个样本集就用该基分类算法在该样本集上产生一个基分类器，这样在给定训练轮数n后,就可产生n个基分类器，然后Boosting框架算法将这n个基分类器进行加权融合，产生一个最后的结果分类器，在这n个基分类器中，每个单个的分类器的识别率不一定很高，但他们联合后的结果有很高的识别率，这样便提高了该弱分类算法的识别率。在产生单个的基分类器时可用相同的分类算法，也可用不同的分类算法，这些算法一般是不稳定的弱分类算法，如神经网络(BP)，决策树(C4.5)等。

# 参考资料

- [百度百科：Boosting](https://baike.baidu.com/item/Boosting/1403912?fr=aladdin)

“Boosting方法概述”这一小节主要参考的就是此文章。