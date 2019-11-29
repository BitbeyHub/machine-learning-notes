# 常见回归和分类损失函数比较

* [返回顶层目录](../../SUMMARY.md)



# 回归问题的损失函数



# 分类问题的损失函数





## 交叉熵损失



### 分类问题中交叉熵优于MSE的原因

<https://zhuanlan.zhihu.com/p/84431551>



## Logistic loss



### Logistic loss和交叉熵损失损失的等价性

对于解决分类问题的FM模型，

当标签为[1, 0]时，其损失函数为交叉熵损失：
$$
Loss=y\ \text{log} \hat{y}+(1-y)\text{log}(1-\hat{y})
$$
当标签为[1, -1]时，其损失函数为
$$
Loss=\text{log}\left(1+\text{exp}(-yf(x))\right)
$$
其中，f(x)是$$w\cdot x$$，不是$$\hat{y}$$。

这两种损失函数其实是完全等价的。

（1）当为正样本时，损失为

- 标签为[1, 0]
  $$
  Loss=-y\text{log}(\hat{y})=-\text{log}\frac{1}{1+\text{exp}(-wx)}=\text{log}(1+\text{exp}(-wx)
  $$

- 标签为[1, -1]
  $$
  Loss=\text{log}\left(1+\text{exp}(-yf(x))\right)=\text{log}\left(1+\text{exp}(-wx)\right)
  $$


（2）当为负样本时，损失为

- 标签为[1, 0]
  $$
  \begin{aligned}
  Loss&=-(1-y)\text{log}(1-\hat{y})=-\text{log}(1-\frac{1}{1+\text{exp}(-wx)})\\
  &=\text{log}(1+\text{exp}(wx))
  \end{aligned}
  $$

- 标签为[1, -1]
  $$
  Loss=\text{log}\left(1+\text{exp}(-yf(x))\right)=\text{log}\left(1+\text{exp}(wx)\right)
  $$


可见，两种损失函数的值完全一样。























# 参考文献

* [常见回归和分类损失函数比较](https://zhuanlan.zhihu.com/p/36431289)

本文参考了此博客。

* [MSE vs 交叉熵](https://zhuanlan.zhihu.com/p/84431551)

"分类问题中交叉熵优于MSE的原因"参考了此博客。

* [Notes on Logistic Loss Function](http://www.hongliangjie.com/wp-content/uploads/2011/10/logistic.pdf)
* [Logistic Loss函数、Logistics回归与极大似然估计](https://www.zybuluo.com/frank-shaw/note/143260)
* [Logistic loss函数](https://buracagyang.github.io/2019/05/29/logistic-loss-function/)

"Logistic loss和交叉熵损失损失的等价性"参考了此博客。