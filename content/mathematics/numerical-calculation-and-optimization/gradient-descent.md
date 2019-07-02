# 梯度下降算法

- [返回顶层目录](../../SUMMARY.md#目录)
- [返回上层目录](numerical-calculation-and-optimization.md)
- [批量梯度下降BGD](#批量梯度下降BGD)
- [随机梯度下降SGD](#随机梯度下降SGD)
  - [大数据中一般梯度下降算法的问题](#大数据中一般梯度下降算法的问题)
  - [随机梯度下降的原理](#随机梯度下降的原理)
  - [随机梯度下降的性质](#随机梯度下降的性质)
  - [SGD过程中的噪声如何帮助避免局部极小值和鞍点](#SGD过程中的噪声如何帮助避免局部极小值和鞍点)
- [小批量梯度下降MBGD](#小批量梯度下降MBGD)
  - [BatchSize的理解](#BatchSize的理解)



[理解梯度下降算法](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247484111&idx=1&sn=4ed4480e849298a0aff828611e18f1a8&chksm=fdb69f58cac1164e844726bd429862eb7b38d22509eb4d1826eb851036460cb7ca5a8de7b9bb&mpshare=1&scene=1&srcid=0511kfAvfTZwwLToaStXHEpp#rd)

[梯度下降法的三种形式BGD、SGD以及MBGD](https://mp.weixin.qq.com/s?__biz=MzA4NzE1NzYyMw==&mid=2247496082&idx=2&sn=e64b40dd4944a3b80b1d3495715536b1&chksm=903f0f8aa748869c1c475da4ea35131bfcf0749abff4697f1ce114ce350b11771569a08b5680&scene=0#rd)

[为什么最速下降法中迭代方向是锯齿形的？](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247486978&idx=1&sn=6ee1f68f96adf7b501389f47796e68ba&chksm=ebb436d6dcc3bfc0d7ea613221a5a79d6561387851b50fb51a96d69e8c661752b6ace1a79e37&scene=0#rd)



# 批量梯度下降BGD





# 随机梯度下降SGD

几乎所有的深度学习算法都用到了随机梯度下降，它是梯度下降算法的拓展。

机器学习中反复出现的一个问题是：好的泛化需要大的训练集，但是，训练集越大，计算代价也越大。

## 大数据中一般梯度下降算法的问题

机器学习算法中，代价函数通常可分解为各样本的代价函数的总和。例如，训练数据的负对数条件似然可写成
$$
J(\theta)=\mathbb{E}_{x,y\sim \hat{p}_{data}}L(x,y,\theta)=\frac{1}{m}\sum^{m}_{i=1}L(x^{(i)},y^{(i)},\theta)
$$
其中，L是每个样本的损失：
$$
L(x,y,\theta)=-log\ p(y|x;\theta)
$$
对于这些相加的代价函数，梯度下降需要计算
$$
\bigtriangledown_{\theta}J(\theta)=\frac{1}{m}\sum^{m}_{i=1}\bigtriangledown_{\theta}L(x^{(i)},y^{(i)},\theta)
$$
这个运算的计算复杂度是O(m)。随着训练集规模增长为数十亿的样本，计算一步梯度也会消耗相当长的时间。

## 随机梯度下降的原理

随机梯度下降的核心是，**梯度是期望。期望可使用小规模的样本近似估计**。

具体步骤：我们在每一步都从训练集中均匀抽出一**小批量（minibatch）**样本
$$
\mathbb{B}=\{x^{(1)},...,x^{(m^{'})}\}
$$
，小批量的数目m'通常是一个相对较小的数，从一到几百。重要的是，当训练集大小m增长时，m'通常是固定的。我们可能在拟合几十亿样本时，每次更新计算只用到几百个样本。

梯度的估计可以表示成
$$
g=\frac{1}{m^{'}}\bigtriangledown_{\theta}\sum^{m^{'}}_{i=1}L(x^{(i)},y^{(i)},\theta)
$$
使用来自小批量$\mathbb{B}$的样本。然后，随机梯度下降算法使用如下的梯度下降算法估计：
$$
\theta=\theta-\epsilon g
$$
其中，$\epsilon$是学习率。

## 随机梯度下降的性质

梯度下降往往被认为很慢或不可靠。

在以前，将梯度下降应用到非凸优化问题被认为很鲁莽或者没有原则。而现在，在深度学习中，使用梯度下降的训练效果很不错。虽然优化算法不一定能保证在合理的时间内达到一个局部最小值，但它通常能及时地找到代价函数一个很小的值，并且是有用的。

在深度学习之外，随机梯度下降有很多重要的应用。它是在大规模数据上训练大型线性模型的主要方法。对于固定大小的模型，每一步随机梯度下降更新的计算量不取决于训练集的大小m。在实践中，当训练集大小增长时，我们通常会使用一个更大的模型，但是这并非是必须的。**达到收敛模型所需的更新次数通常会随着训练集规模增大而增加。然而，当m趋于无穷大时，该模型最终会随机梯度下降抽样完训练集上所有样本之前收敛到可能的最优测试误差。继续增加m不会延长达到模型可能的最优测试误差的时间**。从这点来看，我们可以认为用SGD训练模型的渐进代价是关于m的函数的O(1)级别。

## SGD过程中的噪声如何帮助避免局部极小值和鞍点

https://zhuanlan.zhihu.com/p/36816689





# 小批量梯度下降MBGD

## BatchSize的理解

**直观的理解：** 

Batch Size定义：一次训练所选取的样本数。 

Batch Size的大小影响模型的优化程度和速度。同时其直接影响到GPU内存的使用情况，假如你GPU内存不大，该数值最好设置小一点。

---

**为什么要提出Batch Size？**

在没有使用Batch Size之前，这意味着网络在训练时，是一次把所有的数据（整个数据库）输入网络中，然后计算它们的梯度进行反向传播，由于在计算梯度时使用了整个数据库，所以计算得到的梯度方向更为准确。但在这情况下，计算得到不同梯度值差别巨大，难以使用一个全局的学习率，所以这时一般使用Rprop这种基于梯度符号的训练算法，单独进行梯度更新。 

在小样本数的数据库中，不使用Batch Size是可行的，而且效果也很好。但是一旦是大型的数据库，一次性把所有数据输进网络，肯定会引起内存的爆炸。所以就提出Batch Size的概念。

---

**Batch Size设置合适时的优点：**

1、通过并行化提高内存的利用率。就是尽量让你的GPU满载运行，提高训练速度。 

2、单个epoch的迭代次数减少了，参数的调整也慢了，假如要达到相同的识别精度，需要更多的epoch。 

3、适当Batch Size使得梯度下降方向更加准确。

---

**Batch Size从小到大的变化对网络影响**

1、没有Batch Size，梯度准确，只适用于小样本数据库

2、Batch Size=1，梯度变来变去，非常不准确，网络很难收敛。

3、Batch Size增大，梯度变准确， 

4、Batch Size增大，梯度已经非常准确，再增加Batch Size也没有用

注意：Batch Size增大了，要到达相同的准确度，必须要增大epoch。

---

GD（Gradient Descent）：就是没有利用Batch Size，用基于整个数据库得到梯度，梯度准确，但数据量大时，计算非常耗时，同时神经网络常是非凸的，网络最终可能收敛到初始点附近的局部最优点。

SGD（Stochastic Gradient Descent）：就是Batch Size=1，每次计算一个样本，梯度不准确，所以学习率要降低。

mini-batch SGD：就是选着合适Batch Size的SGD算法，mini-batch利用噪声梯度，一定程度上缓解了GD算法直接掉进初始点附近的局部最优值。同时梯度准确了，学习率要加大。 

对于mini-batch SGD:

* loss值
  $$
  L=\frac{1}{m}\sum_{i=1}^mL(x_i,y_i)
  $$

* gradient值
  $$
  g=\frac{1}{m}\sum_{i=1}^mg(x_i,y_i)
  $$


**为什么说Batch size的增大能使网络的梯度更准确？**

梯度的方差表示： 
$$
\begin{aligned}
Var(g)&=Var\left(\frac{1}{m}\sum_{i=1}^mg(x_i,y_i)\right)\\
&=\frac{1}{m^2}Var\left(g(x_1,y_1)+g(x_2,y_2)+...+g(x_m,y_m)\right)\\
&=\frac{1}{m^2}\left[Var\left(g(x_1,y_1)\right)+Var\left(g(x_1,y_1)\right)+...+Cov\right]\\
&=\frac{1}{m^2}\left(mVar\left(g(x_i,y_i)\right)\right)\\
&=\frac{1}{m}Var\left(g(x_i,y_i)\right)
\end{aligned}
$$
上式第三步是因为：由于样本是随机选取的，满足独立同分布，所以所有样本具有相同的方差。

可以看出当Batch size为m时，样本的方差减少m倍，梯度就更准确了。

假如想要保持原来数据的梯度方差，可以增大学习率lr（learning-rate）。
$$
\frac{1}{m}Var\left(lr\cdot g(x_i,y_i)\right)
$$
只要lr取sqrt(m)，上式就变成Var(g(xi, yi))。

这也说明batch size设置较大时，一般学习率要增大。但是lr的增大不是一开始就设置的很大，而是在训练过程中慢慢变大。

---

一个具体例子分析： 

在分布式训练中，Batch size随着数据并行的workers增加而增大，假如baseline的Batch Size为B，而学习率为lr，训练epoch为N。假如保持baseline的lr，一般达不到很好的收敛速度和精度。 

原因：对于收敛速度，假如有K个workers，则每个批次为KB，因此一个epoch迭代的次数为baseline的1k，而学习率lr不变，所以要达到与baseline相同的收敛情况，epoch要增大。而根据上面公式，epoch最大需要增大KN个epoch，但一般情况下不需要增大那么多。 

对于收敛精度，由于Batch size的使用使梯度更准确，噪声减少，所以更容易收敛。

# 参考资料

- 《深度学习》Goodfellow

“随机梯度下降SGD”参考了此书的第五章5.9小节“随机梯度下降”。

* [神经网络中Batch Size的理解](https://blog.csdn.net/qq_34886403/article/details/82558399)

“BatchSize的理解”参考了此博客。

