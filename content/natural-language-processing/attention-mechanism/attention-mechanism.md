# 注意力机制

* [返回顶层目录](../../SUMMARY.md#目录)



Attention[Attention和Transformer](https://zhuanlan.zhihu.com/p/38485843)

同样是在MT问题中，在seq2seq的基础上提出了attention机制（Bahadanau attention）[（论文地址）](https://arxiv.org/abs/1409.0473)。现在由于性能相对没有attention的原始模型太优越，现在提到的seq2seq一般都指加入了attention机制的模型。同样上面的问题，通过encoder，把 ![[公式]](https://www.zhihu.com/equation?tex=X%3D%28x_0%2Cx_1%2Cx_2%2Cx_3%29) 映射为一个隐层状态 ![[公式]](https://www.zhihu.com/equation?tex=H%3D%28h_0%2Ch_1%2Ch_2%2Ch_3%29) ，再经由decoder将 ![[公式]](https://www.zhihu.com/equation?tex=H%3D%28h_0%2Ch_1%2Ch_2%2Ch_3%29) 映射为 ![[公式]](https://www.zhihu.com/equation?tex=Y%3D%28y_0%2Cy_1%2Cy_2%29) 。这里精妙的地方就在于，Y中的每一个元素都与H中的所有元素相连，而**每个元素通过不同的权值给与Y不同的贡献**。







- [完全图解RNN、RNN变体、Seq2Seq、Attention机制](https://zhuanlan.zhihu.com/p/28054589)





===



[翻译系统/聊天机器人Seq2Seq模型+attention](https://blog.csdn.net/weixin_37479258/article/details/99887469)

**经典篇：《Neural Machine Translation by Jointly Learning to Align and Translate》**     

Attention机制最初由图像处理领域提出，后来被引入到NLP领域用于解决机器翻译的问题，使得机器翻译的效果得到了显著的提升。attention是近几年NLP领域最重要的亮点之一，后续的Transformer和Bert都是基于attention机制。



[注意力机制（Attention Mechanism）](https://blog.csdn.net/yimingsilence/article/details/79208092)



[动画图解Attention机制，让你一看就明白](https://mp.weixin.qq.com/s/-XJeyK6OvjAjDOcpXE7olQ)



[真正的完全图解Seq2Seq Attention模型](https://zhuanlan.zhihu.com/p/40920384?utm_source=wechat_session&utm_medium=social&utm_oi=903049909593317376)

[Attention机制详解（一）——Seq2Seq中的Attention](https://zhuanlan.zhihu.com/p/47063917?utm_source=wechat_session&utm_medium=social&utm_oi=903049909593317376)

[计算机视觉中attention机制的理解](https://zhuanlan.zhihu.com/p/61440116?utm_source=wechat_session&utm_medium=social&utm_oi=903049909593317376)

[Attention！注意力机制模型最新综述（附下载）](https://mp.weixin.qq.com/s/CrxbmG7mbsmERMLEDkGYxw)

[遍地开花的 Attention，你真的懂吗？](https://zhuanlan.zhihu.com/p/77307258)

