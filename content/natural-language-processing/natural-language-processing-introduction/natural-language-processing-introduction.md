# 自然语言处理概论

* [返回顶层目录](../../../SUMMARY.md)





# NLP的发展

我们抽取了三篇论文讲述词向量的发展，一脉相承，从经典到前沿。

**经典篇：《Efficient Estimation of Word Representations in Vector Space》** 

word2vec是将词汇向量化，这样我们就可以进行定量的分析，分析词与词之间的关系，这是one-hot encoding做不到的。Google的Tomas Mikolov 在2013年发表的这篇论文给自然语言处理领域带来了新的巨大变革，提出的两个模型CBOW (Continuous Bag-of-Words Model)和Skip-gram (Continuous Skip-gram Model)，创造性的用预测的方式解决自然语言处理的问题，而不是传统的词频的方法。**奠定了后续NLP处理的基石**。并将NLP的研究热度推升到了一个新的高度。 

**经典篇：《Neural Machine Translation by Jointly Learning to Align and Translate》**     

Attention机制最初由图像处理领域提出，后来被引入到NLP领域用于解决机器翻译的问题，使得机器翻译的效果得到了显著的提升。attention是近几年NLP领域最重要的亮点之一，后续的Transformer和Bert都是基于attention机制。

**经典篇：《Transformer: attention is all you need》** 

这是谷歌与多伦多大学等高校合作发表的论文，提出了一种新的网络框架Transformer，是一种新的编码解码器，与LSTM地位相当。

Transformer是完全基于注意力机制（attention mechanism)的网络框架，使得机器翻译的效果进一步提升，为Bert的提出奠定了基础。该论文2017年发表后引用已经达到1280，GitHub上面第三方复现的star2300余次。可以说是**近年NLP界最有影响力的工作，NLP研究人员必看！**





1

[真正的完全图解Seq2Seq Attention模型](https://zhuanlan.zhihu.com/p/40920384)



[经典算法·从seq2seq、attention到transformer](https://zhuanlan.zhihu.com/p/54368798)









