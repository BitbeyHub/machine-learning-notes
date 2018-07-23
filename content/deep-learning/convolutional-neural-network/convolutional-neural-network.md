# CNN卷积神经网络

* [返回顶层目录](../../SUMMARY.md#目录)
* [卷积](#卷积)
* [池化](#池化)






[CNN卷积神经网络分析](https://zhuanlan.zhihu.com/p/36915223)





# 什么是卷积神经网络



看这个

如何理解卷积神经网络（CNN）中的卷积和池化？

题主在学习ML的过程中发现，在CNN的诸多教程与论文当中对卷积和池化的介绍都不如其他方面直观和易于理解，这个领域对我来说一直是一个黑箱，除了能简单掌握Tensorflow和Theano等的接口函数的使用之外，一直无法对这方面有一个系统的了解。
因此希望能更加系统和直观地理解这两个方面。

可以看看这篇文章《[An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)》，感觉讲的还挺详细
翻译版：[[翻译\] 神经网络的直观解释](http://www.hackcv.com/index.php/archives/104/?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io)



[深度卷积神经网络演化历史及结构改进脉络-40页长文全面解读](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247484037&idx=1&sn=13ad0d521b6a3578ff031e14950b41f4&chksm=fdb69f12cac11604a42ccb37913c56001a11c65a8d1125c4a9aeba1aed570a751cb400d276b6&scene=0#rd)

[入门 | 一文看懂卷积神经网络](https://mp.weixin.qq.com/s?__biz=MzA5ODUxOTA5Mg==&mid=2652557859&idx=1&sn=ad930ee2f1225b68750f6ba85d6c54ee&chksm=8b7e2378bc09aa6ebb8fc34fe51c9ffd7d644d98ead2c8de585d1394ff7e86dfc1ad1f4445ad&scene=0#rd)



# 感性认识

http://scs.ryerson.ca/~aharley/vis/conv/

![cnn_num_recognition](pic/cnn_num_recognition.png)



[卷积网络背后的直觉](https://zhuanlan.zhihu.com/p/37657943)



[能否对卷积神经网络工作原理做一个直观的解释？](https://www.zhihu.com/question/39022858)



[机器视角：长文揭秘图像处理和卷积神经网络架构](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650728746&idx=1&sn=61e9cb824501ec7c505eb464e8317915&scene=0#wechat_redirect)



[An Intuitive Explanation of Convolutional](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)



# 卷积神经网络的发展历程

卷积神经网络的发展过程如图所示：

![cnn-history](pic/cnn-history.png)

卷积神经网络发展的起点是神经认知机(neocognitron)模型，当时已经出现了卷积结构。第一个卷积神经网络模型诞生于1989年，其发明人是 LeCun。学习卷积神经网络的读本是 Lecun的论文，在这篇论文里面较为详尽地解释了什么是卷积神经网络，并且阐述了为什么要卷积，为什么要降采样，径向基函数(radial basis function，RBF)怎么用，等等。

1998年 LeCun 提出了 LeNet，但随后卷积神经网络的锋芒逐渐被 SVM 等手工设计的特征的分类器盖过。随着 ReLU 和 Dropout 的提出，以及GPU和大数据带来的历史机遇，卷积神经网络在2012年迎来了历史性突破—AlexNet。

如图所示，AlexNet 之后卷积神经网络的演化过程主要有4个方向的演化：

- 一个是网络加深；
- 二是增强卷积层的功能；
- 三是从分类任务到检测任务；
- 四是增加新的功能模块。

如上图，分别找到各个阶段的几个网络的论文，理解他们的结构和特点之后，在 TensorFlow Models 下，都有对这几个网络的实现。

对着代码理解，并亲自运行。随后在自己的数据集上做一做 finetune，会对今后工业界进行深度学习网络的开发流程有个直观的认识。

下面就简单讲述各个阶段的几个网络的结构及特点。

## 网络加深

### LeNet

> LeNet 的论文《GradientBased Learning Applied to Document  Recognition》
>
> http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
>
> LeCun的LeNet个人网站
>
> http://yann.lecun.com/exdb/lenet/index.html

LeNet包含的组件如下。

- 输入层：32×32
- 卷积层：3个
- 降采样层：2个
- 全连接层：1个
- 输出层（高斯连接）：10个类别（数字0～9的概率）

LeNet的网络结构如图所示：

![LeNet](pic/LeNet.png)

下面就介绍一下各个层的用途及意义。

- 输入层。输入图像尺寸为32×32。这要比MNIST数据集中的字母（28×28）还大，即对图像做了预处理 reshape 操作。这样做的目的是希望潜在的明显特征，如笔画断续、角点，能够出现在最高层特征监测卷积核的中心。
- 卷积层(C1, C3, C5)。卷积运算的主要目的是使原信号特征增强，并且降低噪音。在一个可视化的在线演示示例中，我们可以看出不同的卷积核输出特征映射的不同，如图所示。
- 下采样层(S2, S4)。下采样层主要是想降低网络训练参数及模型的过拟合程度。通常有以下两种方式。
  - 最大池化(max pooling)：在选中区域中找最大的值作为采样后的值。
  - 平均值池化(mean pooling)：把选中的区域中的平均值作为采样后的值。
- 全连接层(F6)。F6是全连接层，计算输入向量和权重向量的点积，再加上一个偏置。随后将其传递给sigmoid函数，产生单元i的一个状态。
- 输出层。输出层由欧式径向基函数(Euclidean radial basis function)单元组成，每个类别(数字的0～9)对应一个径向基函数单元，每个单元有84个输入。也就是说，每个输出 RBF 单元计算输入向量和该类别标记向量之间的欧式距离。距离越远，RBF 输出越大。

经过测试，采用 LeNet，6万张原始图片的数据集，错误率能够降低到0.95%；54万张人工处理的失真数据集合并上6万张原始图片的数据集，错误率能够降低到0.8%。

接着，历史转折发生在2012年，Geoffrey Hinton 和他的学生 Alex Krizhevsky 在 ImageNet 竞赛中一举夺得图像分类的冠军，刷新了图像分类的记录，通过比赛回应了对卷积方法的质疑。比赛中他们所用网络称为 AlexNet。

### AlexNet

AlexNet 在2012年的 ImageNet 图像分类竞赛中，Top-5错误率为15.3%；2011年的冠军是采用基于传统浅层模型方法，Top-5错误率为25.8%。AlexNet 也远远超过2012年竞赛的第二名，错误率为26.2%。AlexNet 的论文详见 Alex Krizhevsky、Ilya Sutskever 和 Geoffrey E．Hinton 的[《ImageNet Classification with Deep Convolutional Neural Networks》](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)。

AlexNet 的结构如图所示。图中明确显示了两个 GPU 之间的职责划分：一个 GPU 运行图中顶部的层次部分，另一个 GPU 运行图中底部的层次部分。GPU 之间仅在某些层互相通信。

![AlexNet](pic/AlexNet.png)

AlexNet 由5个卷积层、5个池化层、3个全连接层，大约5000万个可调参数组成。最后一个全连接层的输出被送到一个1000维的 softmax 层，产生一个覆盖1000类标记的分布。

AlexNet 之所以能够成功，让深度学习卷积的方法重回到人们视野，原因在于使用了如下方法：

- 防止过拟合：Dropout、数据增强(data augmentation)。
- 非线性激活函数：ReLU。
- 大数据训练：120万(百万级)ImageNet图像数据。
- GPU 实现、LRN(local responce normalization)规范化层的使用。

要学习如此多的参数，并且防止过拟合，可以采用两种方法：数据增强和 Dropout。

数据增强：增加训练数据是避免过拟合的好方法，并且能提升算法的准确率。当训练数据有限的时候，可以通过一些变换从已有的训练数据集中生成一些新数据，来扩大训练数据量。通常采用的变形方式以下几种，具体效果如图所示。

- 水平翻转图像(又称反射变化，flip)。
- 从原始图像(大小为256×256)随机地平移变换(crop)出一些图像(如大小为224×224)。
- 给图像增加一些随机的光照(又称光照、彩色变换、颜色抖动)。

Dropout。AlexNet 做的是以0.5的概率将每个隐层神经元的输出设置为0。以这种方式被抑制的神经元既不参与前向传播，也不参与反向传播。因此，每次输入一个样本，就相当于该神经网络尝试了一个新结构，但是所有这些结构之间共享权重。因为神经元不能依赖于其他神经元而存在，所以这种技术降低了神经元复杂的互适应关系。因此，网络需要被迫学习更为健壮的特征，这些特征在结合其他神经元的一些不同随机子集时很有用。如果没有 Dropout，我们的网络会表现出大量的过拟合。Dropout 使收敛所需的迭代次数大致增加了一倍。

Alex 用非线性激活函数 relu 代替了 sigmoid，发现得到的 SGD 的收敛速度会比 sigmoid/tanh 快很多。单个 GTX 580 GPU 只有3 GB 内存，因此在其上训练的数据规模有限。从 AlexNet 结构图可以看出，它将网络分布在两个 GPU 上，并且能够直接从另一个 GPU 的内存中读出和写入，不需要通过主机内存，极大地增加了训练的规模。

## 增强卷积层的功能

### VGGNet

VGGNet 可以看成是加深版本的 AlexNet，参见 Karen Simonyan 和 Andrew Zisserman 的论文[《Very Deep Convolutional Networks for Large-Scale Visual Recognition》](https://arxiv.org/pdf/1409.1556.pdf)。

VGGNet 和下文中要提到的 GoogLeNet 是2014年 ImageNet 竞赛的第二名和第一名，Top-5错误率分别为7.32%和6.66%。VGGNet 也是5个卷积组、2层全连接图像特征、1层全连接分类特征，可以看作和 AlexNet 一样总共8个部分。根据前5个卷积组，VGGNet 论文中给出了 A～E 这5种配置，如图所示。卷积层数从8（A 配置）到16（E 配置）递增。VGGNet 不同于 AlexNet 的地方是：VGGNet 使用的层更多，通常有16～19层，而 AlexNet 只有8层。

![VGGNet](pic/VGGNet.png)

### GoogLeNet

提到 GoogleNet，我们首先要说起 NIN(Network in Network)的思想(详见 Min Lin 和 Qiang Chen 和 Shuicheng Yan 的论文[《Network In Network》](https://arxiv.org/pdf/1312.4400.pdf))，它对传统的卷积方法做了两点改进：将原来的线性卷积层(linear convolution layer)变为多层感知卷积层(multilayer perceptron)；将全连接层的改进为全局平均池化。这使得卷积神经网络向另一个演化分支—增强卷积模块的功能的方向演化，2014年诞生了 GoogLeNet(即 Inception V1)。谷歌公司提出的 GoogLeNet 是2014年 ILSVRC 挑战赛的冠军，它将 Top-5的错误率降低到了6.67%。GoogLeNet 的更多内容详见 Christian Szegedy 和 Wei Liu 等人的论文[《Going Deeper with Convolutions》](https://arxiv.org/pdf/1409.4842.pdf)。

GoogLeNet 的主要思想是围绕“深度”和“宽度”去实现的。

- 深度。层数更深，论文中采用了22层。为了避免梯度消失问题，GoogLeNet 巧妙地在不同深度处增加了两个损失函数来避免反向传播时梯度消失的现象。
- 宽度。增加了多种大小的卷积核，如1×1、3×3、5×5，但并没有将这些全都用在特征映射上，都结合起来的特征映射厚度将会很大。但是采用了[论文](https://arxiv.org/pdf/1409.4842.pdf)中Figure2右侧所示的降维的 Inception 模型，在3×3、5×5卷积前，和最大池化后都分别加上了1×1的卷积核，起到了降低特征映射厚度的作用。



# 卷积

一文全解深度学习中的卷积

https://zhuanlan.zhihu.com/p/36742352

[卷积特征提取](http://deeplearning.stanford.edu/wiki/index.php/%E5%8D%B7%E7%A7%AF%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96)

CNN（卷积神经网络），它的基本假设是特征的不同维度之间有局部相关性，卷积操作可以抓住这只局部相关性，形成新的特征。比如自然语言里，有重复出现的bigram，或者图像里代表性的局部像素块。不满足这种局部相关性的数据，比如收到的邮件序列，这种局部相关性很弱，那用CNN就不能抓到有用的特征。作者：李韶华链接：https://www.zhihu.com/question/46301335/answer/112354887来源：知乎著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。





[哪位高手能解释一下卷积神经网络的卷积核？](https://www.zhihu.com/question/52237725/answer/172479030)

**Convolution Kernel 的意义**

Convolution Kernel 其实在图像处理中并不是新事物，Sobel 算子等一系列滤波算子，一直都在被用于边缘检测等工作中，只是以前被称为 Filter。做图像处理的同学应该有印象。

Convolution Kernel 具有的一个属性就是局部性。即它只关注局部特征，局部的程度取决于 Convolution Kernel 的大小。比如用 Sobel 算子进行边缘检测，本质就是比较图像邻近像素的相似性。

也可以从另外一个角度理解 Convolution Kernel 的意义。学过信号处理的同学应该记得，时域卷积对应频域相乘。所以原图像与 Convolution Kernel 的卷积，其实是对频域信息进行选择。比如，图像中的边缘和轮廓属于是高频信息，图像中某区域强度的综合考量属于低频信息。在传统图像处理里，这是指导设计 Convolution Kernel 的一个重要方面。





# 池化

[池化](http://deeplearning.stanford.edu/wiki/index.php/%E6%B1%A0%E5%8C%96)



池化层是干什么的呢？池化，英文是pooling，字面上看挺难懂，但其实这可能是CNN里最简单的一步了。我们可以不按字面理解，把它理解成下采样（subsampling）。池化分为最大值池化和平均值池化，和卷积差不多，也是取一小块区域，比如一个5*5的方块，如果是最大值池化，那就选这25个像素点最大的那个输出，如果是平均值池化，就把25个像素点取平均输出。

这样做有什么好处呢？1、应该很明显可以看出，图像经过了下采样尺寸缩小了，按上面的例子，原来5*5的一个区域，现在只要一个值就表示出来了！2、增强了旋转不变性，池化操作可以看做是一种强制性的模糊策略，举个不恰当的例子，假设一个猫的图片，猫的耳朵应该处于左上5x5的一个小区域内（实际上猫的耳朵不可能这么小），无论这个耳朵如何旋转，经过池化之后结果都是近似的，因为就是这一块取平均和取最大对旋转并不care。

当然和之前的卷积一样，池化也是层层递进的，底层的池化是在模糊底层特征，如线条等，高层的池化模糊了高级语义特征，如猫耳朵。所以，一般的CNN架构都是三明治一样，卷积池化交替出现，保证提取特征的同时也强制模糊增加特征的旋转不变性。



# 损失函数



贝叶斯眼里的正则化

https://zhuanlan.zhihu.com/p/32685118

总结一下，交叉熵损失函数（cross entropy loss），其本质就是最大似然估计MLE，而正则化等价于MLE加上**先验分布**。所以，从贝叶斯角度来看，损失函数＋正则化就是贝叶斯最大后验估计MAP。



# CNN训练原理

《[Notes on Convolutional Neural Networks](http://web.mit.edu/jvb/www/papers/cnn_tutorial.pdf)》，这是Jake Bouvrie在2006年写的关于CNN的训练原理，虽然文献老了点，不过对理解经典CNN的训练过程还是很有帮助的。该作者是剑桥的研究认知科学的。本文参照了这篇翻译《[Notes on Convolutional Neural Networks翻译](https://blog.csdn.net/zouxy09/article/details/9993371)》，并在此基础上增加了自己的修改。

## 0概要

主要讲了CNN的Feedforward Pass和 Backpropagation Pass，关键是卷积层和polling层的BP推导讲解。

## 1引言

这个文档是为了讨论CNN的推导和执行步骤的，并加上一些简单的扩展。因为CNN包含着比权重还多的连接，所以结构本身就相当于实现了一种形式的正则化了。另外CNN本身因为结构的关系，也具有某种程度上的平移不变性。这种特别的NN可以被认为是以数据驱动的形式在输入中可以自动学习过滤器来自动的提取特征。我们这里提出的推导是具体指2D数据和卷积的，但是也可以无障碍扩展到任意维度上。

我们首先以在全连接网络上说明经典的Bp是如何工作的，然后介绍了在2DCNN中BP是如何在过滤器和子采样层上进行权值更新的。通过这些论述，我们强调了模型实际执行的高效的重要性，并给出一小段MATLAB代码来辅助说明这些式子。当然在CNN上也不能过度的夸大高效代码的重要性（毕竟结构放在那里了，就是很慢的）。接下来就是讨论关于如何将前层学到的特征图自动组合的主题，并具体的考虑学习特征图的稀疏组合问题。

免责声明：这个粗糙的笔记可能包含错误，各位看官且看且谨慎。（作者的免责声明）。

## 2用BP训练全连接网络

​     在许多文献中，可以发现经典的CNN是由卷积和子采样操作互相交替组成的，然后在最后加上一个普通的多层网络的结构：最后几层（最靠经输出层的部分）是全连接1D层。当准备好将最后的2D特征图作为输入馈送到这个全连接1D网络的时候，也能很方便的将所有的输出图中表现的特征连接到一个长输入向量中，并往回使用BP进行训练。这个标准的BP算法将会在具体介绍CNN的情况之前介绍（【1】中有更详细的介绍）。

### 2.1前向传播

​        在推导过程中，我们的损失函数采用的是误差平方和损失函数。对于一个有着c个类别和N个训练样本的多类问题，这个损失函数形式如下：

式(1)：
$$
E^N=\frac{1}{2}\sum_{n=1}^N\sum_{k=1}^c(t_k^n-y_k^n)^2
$$
这里$t_k^n$是第n个样本相对应的目标（标签）的第k维，$y_k^n$是由模型的第n个样本预测得到的目标（标签）的第k维。对于多分类问题，这个目标通常是以“one-of-c”编码的形式存在的，当$x^n$是属于第k类的，那么$t^n$的第k个元素就是正的，其他的元素就是0或者是负的（这取决于激活函数的选择）。

 因为在整个训练集上的误差只是简单的将每个样本产生的误差进行相加得到的，所以这里先在单个样本（第n 个）上用BP来做讲解：

式(2)：
$$
E^n=\frac{1}{2}\sum_{k=1}^c(t_k^n-y_k^n)^2=\frac{1}{2}\left \| t^n-y^n \right \|_2^2
$$
在普通的全连接网络上，我们能够用下面的BP规则的形式来对 E求关于权重的偏导。 这里$l$指示当前的第几层，输出层为第L层，而输入层（原始数据层）为第1层。这里第$l$层（当前层）的输出是：

式(3)：
$$
x^l=f(u^l),\quad \text{with } u^l=W^lx^{l-1}+b^l
$$
这里，当前层的W是指当前层输入一侧的权值，而不是当前层输出一侧的权值（那是下一层的W了）。这里输出的激活函数 f（**·**）通常是选择逻辑（sigmoid）函数
$$
f(x)=\frac{1}{1+e^{-\beta x}}
$$
或者双曲线tangent函数f(x)=a\*tanh(b\*x)。这个逻辑函数可以将[-∞，+∞]的数映射到[0，1]，而这个双曲线tangent函数可以将[-∞，+∞]的数映射到[-a，+a]。因此双曲线tangent函数的输出通常是靠近0 ，而sigmoid函数的输出通常是非0的。然而对训练数据进行归一化到0均值和单位方差（方差为1）可以在梯度下降上改善收敛。在基于一个归一化的数据集上，通常更喜欢选择双曲线tangent函数。LeCun建议 a=1.7159；b=2/3。这样非线性最大化的点会出现在f(±1)=±1，因此当期望的训练目标以值±1进行归一化的时候，就可以可以避免在训练的时候饱和的问题（估计就是防止训练目标分布的太集中在0周围了，这样可以使它们更加合理的分布）。

### 2.2后向传播

​        网络中我们需要后向传播的“ 误差”可以被认为是关于有偏置项扰动的每个单元的 “敏感性”（这个解释来自于Sebastian Seung）。也就是说：

式(4)：
$$
\frac{\partial E}{\partial b}=\frac{\partial E}{\partial u}\frac{\partial u}{\partial b}=\delta
$$
这里，u是当前层的输入，具体如公式3所示。

式(3)：
$$
x^l=f(u^l),\quad \text{with } u^l=W^lx^{l-1}+b^l
$$
因为
$$
\frac{\partial u}{\partial b}=1
$$
，所以偏置的敏感性其实等于一个单元的所有输入产生的误差偏导。下面的就是从高层到低层的BP：

式(5)：
$$
\delta_l=\left(W^{l+1}\right)^T\left[\delta_{l+1}\ \text{o}\ f'(u^l)\right]
$$
这里的“o” 表示是逐元素相乘的，相当于matlab中的矩阵按各元素位置的点乘：C=A.*B。

上式怎么理解呢？上式的左边是当前层l的输入u对误差的偏导，这是BP算法的难点，但是这个如果对BP算法有所了解的话应该很好理解。我们对照着下图来理解公式5

![two-layers-neural-network](pic/two-layers-neural-network.png)

假设l+1层有m和神经元，l层有n个神经元，我们先对第l层的第h个神经元的输入求导，这个只要搞懂了，然后就很容易扩展为对第l层整个n个神经元求导了。

先对第l层的第h个神经元的输入求导：
$$
\begin{aligned}
\delta_h^l=\frac{\partial E}{\partial u_h^l}&=\frac{\partial E}{\partial x_h^l}\cdot \frac{\partial x_h^l}{\partial u_h^l}\\
&=\left[\sum_{j=1}^m\left(\frac{\partial E}{\partial u_j^{l+1}}\cdot \frac{\partial u_j^{l+1}}{\partial x_h^l}\right)\right]f'(u^l_h)\\
&=\left[\sum_{j=1}^m\left(\delta_j^{l+1}\cdot W_{hj}^{l+1}\right)\right]f'(u^l_h)\\
&=
\begin{bmatrix}
W_{h1}^{l+1} & W_{h2}^{l+1}  & ... & W_{hj}^{l+1}  & ... & W_{hm}^{l+1}
\end{bmatrix}\begin{bmatrix}
\delta_1^{l+1}\\
\delta_2^{l+1}\\
...\\
\delta_j^{l+1}\\
...\\
\delta_m^{l+1}\\
\end{bmatrix}
f'(u^l_h)
\end{aligned}
$$
然后容易扩展为对第l层整个n个神经元求导：
$$
\begin{aligned}
\delta^l=\frac{\partial E}{\partial u^l}
&=
\begin{bmatrix}
\delta_1^l\\ \delta_2^l\\ ... \\ \delta_h^l\\ ...\\ \delta_n^l
\end{bmatrix}\\
&=
\begin{bmatrix}
W_{11}^{l+1} & W_{12}^{l+1}  & ... & W_{1j}^{l+1}  & ... & W_{1m}^{l+1}\\
W_{21}^{l+1} & W_{22}^{l+1}  & ... & W_{2j}^{l+1}  & ... & W_{2m}^{l+1}\\
...& ...& ...& ...& ...& ...\\ 
W_{h1}^{l+1} & W_{h2}^{l+1}  & ... & W_{hj}^{l+1}  & ... & W_{hm}^{l+1}\\
...& ...& ...& ...& ...& ...\\ 
W_{n1}^{l+1} & W_{n2}^{l+1}  & ... & W_{nj}^{l+1}  & ... & W_{nm}^{l+1}\\
\end{bmatrix}\begin{bmatrix}
\delta_1^{l+1}\\
\delta_2^{l+1}\\
...\\
\delta_j^{l+1}\\
...\\
\delta_m^{l+1}\\
\end{bmatrix}
\text{o}
\begin{bmatrix}
f'(u^l_1)\\ 
f'(u^l_2)\\ 
...\\
f'(u^l_h)\\
...\\
f'(u^l_n)
\end{bmatrix}\\
&=\left(W^{l+1}\right)^T\delta^{l+1}\ \text{o}\ f'(u^l)
\end{aligned}
$$


其中，
$$
\begin{aligned}
&W^{l+1}\text{为}[m\times n]\text{维向量,}\text{是}l+1\text{层神经元的输入权重,}m\text{是}l+1\text{层神经元的个数,}n\text{是}l\text{层神经元的个数;}\\
&\delta^{l+1}\text{为}[m\times 1]\text{维向量,}m\text{是}l+1\text{层神经元的个数;}\\
&f'(u^l)\text{是}[n\times 1]\text{维向量了}\\
\end{aligned}
$$

对于公式2中的误差函数，输出层神经元的敏感性如下：

式(6)：
$$
\delta^L=f'(u^L)\ \text{o}\ (y^n-t^n)
$$
最后，关于一个指定的神经元的权重的更新的求导规则就是该神经元的输入乘上该神经元的δ进行缩放罢了（其实就是如下面公式7的两个相乘而已）。在向量的形式中，这相当于输入向量（前层的输出）和敏感性向量的外积：

因为如公式3所示：
$$
u^l=W^lx^{l-1}+b^l
$$
所以有：

式(7)：
$$
\begin{aligned}
\frac{\partial E}{\partial W^l}&=\frac{\partial E}{\partial u^l}\frac{\partial u^l}{\partial x^{l-1}}=x^{l-1}(\delta^l)^T\\
&=\begin{bmatrix}
x_1^{l-1} \\ x_2^{l-1} \\ ... \\ x_h^{l-1} \\ ... \\ x_n^{l-1}
\end{bmatrix}\begin{bmatrix}
\delta_1^l & \delta_2^l & ... & \delta_j^l & ... & \delta_m^l
\end{bmatrix}\\
&=\begin{bmatrix}
x_{1}^{l-1}\delta_1^l & x_{1}^{l-1}\delta_2^l  & ... & x_{1}^{l-1}\delta_j^l  & ... & x_{1}^{l-1}\delta_m^l\\
x_{2}^{l-1}\delta_1^l & x_{2}^{l-1}\delta_2^l & ... & x_{2}^{l-1}\delta_j^l  & ... & x_{2}^{l-1}\delta_m^l\\
...& ...& ...& ...& ...& ...\\ 
x_{h}^{l-1}\delta_1^l & x_{h}^{l-1}\delta_2^l  & ... & x_{h}^{l-1}\delta_j^l  & ... & x_{h}^{l-1}\delta_m^l\\
...& ...& ...& ...& ...& ...\\ 
x_{n}^{l-1}\delta_1^l & x_{n}^{l-1}\delta_2^l  & ... & x_{n}^{l-1}\delta_j^l  & ... & x_{n}^{l-1}\delta_m^l\\
\end{bmatrix}
\end{aligned}
$$

式(8)：
$$
\Delta W^l=-\eta\frac{\partial E}{\partial W^l}
$$

和公式4的偏置更新的表现形式相类似。在实际操作中这里的学习率一般是每个权重都有不同的学习率$η_{ij}$。

## 3CNN

通常卷积层都是有子采样层的附加以此来减少计算时间并且逐步的建立更深远的空间和构型的不变性。在照顾特异性的同时也需要一个小的子采样因子，当然这个不是新方法，但是这个概念却简单而又有效。哺乳动物的视觉皮层和在【12 8 7】中的模型着重的介绍了这些方面，在过去的10年中听觉神经科学的发展让我们知道在不同动物的皮层上primary和belt听觉领域中有相同的设计模型【6 11 9】。层级分析和学习结构也许就是听觉领域获得成功的关键。

### 3.1卷积层

这里接着谈论网络中关于卷积层的BP更新。在一个卷积层中，前层的特征映射图是先进行卷积核运算然后再放入一个激活函数来得到特征映射图作为输出。每个输出图也许是有许多个图的卷积组合而成的。通常如下面的公式：

式(9)：
$$
X_j^l=f\left(\sum_{i\in M_j}x_i^{l-1}*k_{ij}^l+b_j^l\right)
$$
这里$M_j$表示选择的输入图的集合，在MATLAB中这里的卷积是“valid”边缘处理的。那么到底选择哪些输入图呢？通常对输入图的选择包括选择一对或者是三个的，但是下面会讨论如何去自动选择需要组合的特征图。每个输出图都有个额外的偏置b，然而对于一个特定的输出图来说，卷积每个输入图的卷积核是不一样的。也就是说如果输出图j和k都是在输入图i上相加得到的，输出图j应用在图i上的卷积核是和输出图k应用在图i上的卷积核是不同的。

#### 3.1.1计算梯度

我们假设每个卷积层l后面都跟着一个下采样层l+1。在BP算法中，根据上文我们知道，要想求得层l的每个神经元对应的权值的权值更新，就需要先计算与当前层中这个单元相关联的下一层的敏感性$δ_h^{l+1}$，并分别乘以对应的下一层（第l+1层）与这个单元之间的权重参数$W_h^{l+1}$，其总和就是传递到这一层这个单元的敏感性。并用当前层l的该神经元节点h的输入$u_h^l$的激活函数f的偏导乘以这个量就是第l层中当前单元h的敏感性$δ_h^l$了：
$$
\delta_h^l=\left(W_h^{l+1}\right)^T\delta_h^{l+1}\text{o}\ f'(u_h^l)
$$
（这里所要表达的就是BP的想法，如果熟悉BP那么这里比较绕口的意思就能很好的明白了）。

在一个卷积层并后面跟个下采样层的情况中，在下采样层中所关联的图 中的一个像素相当于卷积层输出图中的像素块（后来的池化的想法）。因此在第 层中的一个图上的每个单元只与第 +1层中相对应的图中的一个单元相连（多对一的关系）。为了高效的计算第层的敏感性，我们可以对下采样层中的敏感图进行上采样去保证与卷积层图具有相同的尺寸，随后只需要将第 +1层中的上采样敏感图与第层中的激活函数偏导图逐元素相乘即可。在下采样层图中定义的“权重”全等于  （一个常量，见部分3.2），所以只是将前面步骤的结果用来进行缩放以达到计算 的结果。我们能在卷积层的每个图 j 中进行同样的计算，并用子采样层中相对应的图进行搭配：

### 3.2子采样层



#### 3.2.1计算梯度



### 3.3学习特征图的组合



#### 3.3.1采用稀疏组合



### 3.4在Matlab中的加速



## 4实际训练问题（不完整）



### 4.1批量更新VS在线更新



### 4.2学习率



### 4.3损失函数的选择



### 4.4检查求导是否正确







# 预防过拟合



[激活引入非线性，池化预防过拟合（上）](https://zhuanlan.zhihu.com/p/32793922)



# 参考资料

* [行家 | 如何跨领域成为一位人工智能工程师？](https://blog.csdn.net/u4110122855/article/details/78043171)

"卷积神经网络的发展历程"一节参考了该文。

