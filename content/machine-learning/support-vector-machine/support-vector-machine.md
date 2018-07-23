# 支持向量机

* [返回顶层目录](../../SUMMARY.md#目录)




这份材料从前几节讲的logistic回归出发，引出了SVM，既揭示了模型间的联系，也让人觉得过渡更自然。

https://www.cnblogs.com/jerrylead/archive/2011/03/13/1982639.html





攀登传统机器学习的珠峰-SVM (上)

https://zhuanlan.zhihu.com/p/36332083



理解SVM的核函数和参数

https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247484495&idx=1&sn=4f3a6ce21cdd1a048e402ed05c9ead91&chksm=fdb699d8cac110ce53f4fc5e417e107f839059cb76d3cbf640c6f56620f90f8fb4e7f6ee02f9&mpshare=1&scene=1&srcid=0522xo5euTGK36CZeLB03YGi#rd



[零基础学SVM—Support Vector Machine(一)](https://zhuanlan.zhihu.com/p/24638007)



[SVM原理及推导](https://mp.weixin.qq.com/s?__biz=MzU1NTUxNTM0Mg==&mid=2247488814&idx=1&sn=8d48873a2ac4d41fe0bdb1809865bed7&chksm=fbd2798fcca5f099906c24fa09c6574a0ede8b963638fea62abcc1fb33a6502cdfe423ce163d&scene=0#rd)





# SVM的正则化

支持向量机SVM优化目的为寻找一个超平面，使得正负样本能够以最大间隔分离开，从而得到更好的泛化性能，其通过引入核函数来将低维线性不可分的样本映射到高维空间从而线性可分，通过引入惩罚参数C(类似于正则化参数)来对错分样本进行惩罚，从而减少模型复杂度，提高泛化能力，其优化目标如下：
$$
\mathop{min}_{\theta,b}\frac{1}{n}\sum_{i=1}^n\text{max}(1-y_i(\theta^Tx_i+b),0)+\frac{1}{2CN}\theta^T\theta
$$
其中，正则项系数为
$$
\lambda=\frac{1}{2C}
$$
惩罚参数C作用和正则化参数λ作用一致，只是反相关而已。

需要明白以下结论：

* C越大，λ越小，表示对分错样本的惩罚程度越大，正则化作用越小，偏差越小，方差越大，越容易出现过拟合(通俗理解，原本将低维空间映射到5维空间正好线性可分，但是由于惩罚过于严重，任何一个样本分错了都不可原谅，结果系统只能不断提高维数来拟合样本，假设为10维，最终导致映射维数过高，出现过拟合样本现象，数学上称为VC维较大)；
* C越小，λ越大，表示对分错样本的惩罚程度越小，正则化作用越大，偏差越大，方差越小，越容易出现欠拟合(通俗理解，原本将低维空间映射到5维空间正好线性可分，但是由于惩罚过小，分错了好多样本都可以理解，比较随意，结果系统也采用简化版来拟合样本，假设为3维，最终导致映射维数过低，出现欠拟合样本现象，数学上称为VC维较小)。


# 逻辑回归和SVM的区别

[逻辑回归和SVM的区别是什么？各适用于解决什么问题？](https://www.zhihu.com/question/24904422)



# 参考资料

* [史上最全面的正则化技术总结与分析--part2](https://zhuanlan.zhihu.com/p/35432128)

“SVM的正则化”参考此知乎专栏文章。

