# Transformer

* [返回顶层目录](../../../SUMMARY.md)



[Attention和Transformer](https://zhuanlan.zhihu.com/p/38485843)

Transformer是Attention is all you need[这篇论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.03762)里提出的一个新框架。因为最近在MSRA做时序相关的研究工作，我决定写一篇总结。本文主要讲一下Transformer的网络结构，因为这个对RNN的革新工作确实和之前的模型结构相差很大，而且听我的mentor Shujie Liu老师说在MT（machine translation）领域，transformer已经几乎全面取代RNN了。包括前几天有一篇做SR（speech recognition）的工作也用了transformer，而且SAGAN也是……总之transformer确实是一个非常有效且应用广泛的结构，应该可以算是即seq2seq之后又一次“革命”。





**经典篇：《Transformer: attention is all you need》** 

这是谷歌与多伦多大学等高校合作发表的论文，提出了一种新的网络框架Transformer，是一种新的编码解码器，与LSTM地位相当。

Transformer是完全基于注意力机制（attention mechanism)的网络框架，使得机器翻译的效果进一步提升，为Bert的提出奠定了基础。该论文2017年发表后引用已经达到1280，GitHub上面第三方复现的star2300余次。可以说是**近年NLP界最有影响力的工作，NLP研究人员必看！**



[经典算法·从seq2seq、attention到transformer](https://zhuanlan.zhihu.com/p/54368798)

[通俗易懂！使用Excel和TF实现Transformer](https://mp.weixin.qq.com/s/QRiNGxA-D9MLvv8GPnlhHg)

[搞懂Transformer结构，看这篇PyTorch实现就够了（上）](https://zhuanlan.zhihu.com/p/48731949)



===

[深度学习：transformer模型](https://blog.csdn.net/pipisorry/article/details/84946653)

[Transformers 研究指南](https://mp.weixin.qq.com/s?src=11×tamp=1571917795&ver=1932&signature=DLFKFrYQf7TOR7MXG3wvOYvj0ohuNjLUhNG8AQyEEveK3Zs-vCzNZvbl3*KNKIOLGerlgfPcfpfRylMzxUi*wIafeZDU3J9b0ARWA1vuYxHMGDaI3lEE9a*bTQonVDeo&new=1)

阿里杭州发的deep session interest nerwork DSIN 使用transformer还是有一些道理的，bert4rec相对就……

[BERT4REC：使用Bert进行推荐](https://zhuanlan.zhihu.com/p/97123417)



作者采用Attention机制的原因是考虑到RNN（或者LSTM，GRU等）的计算限制为是有顺序的，也就是说RNN相关算法只能从做向右依次计算或者从右向左依次计算，这种机制带来了两个问题：

1、时间片t的计算以来t-1时刻的计算结果，这样限制了模型的并行能力；

2、顺序计算的过程中信息会丢失，尽管LSTM等门机制的结构一定程度上缓解了长期依赖的问题，但是对于特别长期的依赖现象，LSTM依旧无能为力。

Transformer的提出解决了上面两个问题，首先它使用了Attention机制，将序列中的任意两个位置之间的距离缩小为了一个常量；其次它不是类似于RNN的顺序结构，因此具有更好的并行性，符合现有的GPU框架。论文中给出Transformer的定义是：Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence aligned RNNs or convolution。



