# Facebook的GBDT+LR

论文地址：http://quinonero.net/Publications/predicting-clicks-facebook.pdf





# 参考资料

[推荐系统遇上深度学习(十)--GBDT+LR融合方案实战](https://www.jianshu.com/p/96173f2c2fb4)





[GBDT+LR算法解析及Python实现](https://www.cnblogs.com/wkang/p/9657032.html)

[gbdt对标量特征要不要onehot编码](https://ask.julyedu.com/question/7720)



在我看来，GBDT中每课tree的作用是进行supervised clustering，最后输出的其实是每个cluster的index。用GBDT是为了追求tree间的diversity。类似思路见周志华的gcForest，用extreme random forest进行特征转换，也是为了追求diversity。

[知乎：LR,gbdt,libfm这三种模型分别适合处理什么类型的特征](https://www.zhihu.com/question/35821566/answer/225927793)





[10分钟了解GBDT＋LR模型的来龙去脉](https://cloud.tencent.com/developer/news/14063)



[XGBoost+LR融合的原理和简单实现](https://zhuanlan.zhihu.com/p/42123341?utm_source=wechat_session&utm_medium=social&utm_oi=903049909593317376)









