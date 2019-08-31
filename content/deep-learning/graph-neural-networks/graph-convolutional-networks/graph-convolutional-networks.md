# GCN图卷积网络

* [返回顶层目录](../../../SUMMARY.md#目录)
* [返回上层目录](../graph-neural-networks.md)
* [GCN图卷积网络初步理解](gcn-preliminary-understand.md)
* [GCN图卷积网络的numpy简单实现](gcn-numpy-fulfillment.md)
* [GCN图卷积网络本质理解](gcn-essential-understand.md)
* [GCN图卷积网络全面理解](gcn-comprehensive-understand.md)



W是普通的MLP. 前乘的$\tilde{A}$矩阵是行变换，是把一个点的所有邻接点特征向量加权平均，赋给该点；后乘的W矩阵是列变换，是每个点特征向量各个维度之间的交互。如果没有前面那个$\tilde{A}$，$\sigma(HW)$就是把所有点的特征放到一个batch里，通过一层全连接网络输出。





# 参考资料



[知乎：如何理解 Graph Convolutional Network（GCN）？](https://www.zhihu.com/question/54504471)

[何时能懂你的心——图卷积神经网络（GCN）](https://mp.weixin.qq.com/s/I3MsVSR0SNIKe-a9WRhGPQ)

[Graph Neural Network：GCN 算法原理，实现和应用](https://mp.weixin.qq.com/s/ftz8E5LffWFfaSuF9uKqZQ)

[从图(Graph)到图卷积(Graph Convolution)：漫谈图神经网络模型 (一)](https://www.cnblogs.com/SivilTaram/p/graph_neural_network_1.html)

[如何理解 Graph Convolutional Network（GCN）？](https://ai.yanxishe.com/page/postDetail/13980?from=timeline)

[入门学习 | 什么是图卷积网络？行为识别领域新星](https://mp.weixin.qq.com/s/5wSgC4pXBfRLoCX-73DLnw)

[图卷积神经网络(GCN)详解:包括了数学基础(傅里叶，拉普拉斯)](https://zhuanlan.zhihu.com/p/67522582)

[论文浅尝 | 图神经网络综述：方法及应用](http://blog.openkg.cn/%E8%AE%BA%E6%96%87%E6%B5%85%E5%B0%9D-%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%BC%E8%BF%B0%EF%BC%9A%E6%96%B9%E6%B3%95%E5%8F%8A%E5%BA%94%E7%94%A8/)



[图卷积神经网络相关资源：附有基于图卷积神经网络的实现、示例和教程；](https://github.com/Jiakui/awesome-gcn)





[20190820近期必读的7篇 IJCAI 2019【图神经网络（GNN）】相关论文](https://mp.weixin.qq.com/s/Mp-iLuPScFjyhq3IwzRGHA)

[20190806近期必读的12篇KDD 2019【图神经网络（GNN）】相关论文](https://mp.weixin.qq.com/s/r1K2Ry_GR1RN0frcr_HzLA)

[【清华NLP】图神经网络GNN论文分门别类，16大应用200+篇论文最新推荐](https://mp.weixin.qq.com/s/NYoObFBacOamjo2KHjJOAg)

[斯坦福大学最新论文|知识图卷积神经网络在推荐系统中的应用](https://mp.weixin.qq.com/s/4KS_HG7rBOQgcTII7YKsaQ)

