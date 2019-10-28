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



