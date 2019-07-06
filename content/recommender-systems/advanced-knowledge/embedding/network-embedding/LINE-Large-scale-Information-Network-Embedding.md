# LINE: Large-scale Information Network Embedding

* [返回顶层目录](../../../../../SUMMARY.md)
* [返回上层目录](../embedding.md)





论文地址：https://arxiv.org/pdf/1503.03578.pdf

代码地址：https://github.com/tangjianpku/LINE



# 论文解读



## 使用Alias-Sampling-Method采样边







# 源码说明与运行

## 代码说明

在<https://github.com/tangjianpku/LINE>下载代码。下载下来的代码说明：

**LINE：大规模的信息网络嵌入**

**介绍**

这是为了嵌入非常大规模的信息网络而开发的LINE工具箱。它适用于各种网络，包括有向，无向，无权或加权边。 LINE模型非常高效，能够在几个小时内在单台机器上嵌入数百万个顶点和数十亿个边界的网络。

联系人：唐建，tangjianpku@gmail.com

项目页面：https://sites.google.com/site/pkujiantang/line

当作者在微软研究院工作时，这项工作就完成了

**用法**

我们提供Windows和LINUX版本。为了编译源代码，需要一些外部包，用于为LINE模型中的边缘采样算法生成随机数。对于Windows版本，使用BOOST软件包，可以从http://www.boost.org/下载;对于LINUX，使用GSL包，可以从http://www.gnu.org/software/gsl/下载。

**网络输入**

网络的输入由网络中的边组成。输入文件的每一行代表网络中的一个DIRECTED边缘，指定为格式“起点-终点-权重”（可以用空格或制表符分隔）。对于每个无向边，用户必须使用两个DIRECTED边来表示它。以下是一个词共现网络的输入示例：

```shell
good the 3
the good 3
good bad 1
bad good 1
bad of 4
of bad 4
```

**运行**

```shell
./line -train network_file -output embedding_file -binary 1 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20
```

* train，网络的输入文件;
* output，嵌入的输出文件;
* binary，是否以二进制模式保存输出文件;默认是0（关）;
* size，嵌入的维度;默认是100;
* order，使用的相似度; 1为一阶，2为二阶;默认是2;
* negative，负采样中负采样样本的数目；默认是5;
* samples，训练样本总数（*百万）;
* rho，学习率的起始值;默认是0.025;
* threads，使用的线程总数;默认是1。

**文件夹中的文件**

reconstruct.cpp：用于将稀疏网络重建为密集网络的代码

line.cpp：LINE的源代码;

normalize.cpp：用于归一化嵌入的代码（l2归一化）;

concatenate.cpp：用于连接一阶和二阶嵌入的代码;

**示例**

运行Youtube数据集（可在http://socialnetworks.mpi-sws.mpg.de/data/youtube-links.txt.gz处获得）的示例在train_youtube.bat / train_youtube .sh文件中提供

## 数据集

Youtube数据集：网络中的节点代表用户，有联系的用户之间有边。YouTube网络是一个无向、无权的网络。

数据集从http://socialnetworks.mpi-sws.mpg.de/data/youtube-links.txt.gz下载。下载的数据集文件中，每行有两个数字，中间以制表符隔开，代表网络中的一条边，两个数字分别代表边的起点和终点。因为是无权图，因此不需要权重的值。因为是无向图，因此每条边在文件中出现两次，如1 2和2 1，代表同一条边。

数据集中共包括4945382条边（有向边，因为无向图中每条边被看做两条有向边，所以Youtube网络中有2472691条边）和至少937968个点（文件中节点的名字并不是连续的，有些节点的度为0，在数据集文件中没有出现）。

## 运行示例

在Youtube数据集上运行算法的示例在train_youtube.bat / train_youtube.sh文件中提供。算法运行分为五步：

* 将单向的关系变为双向的关系，因为youtobe好有关系是无向图

  ```shell
  python3 preprocess_youtube.py youtube-links.txt net_youtube.txt
  ```

* 通过reconstruct程序对原网络进行重建（1h）

  ```shell
  ./reconstruct -train net_youtube.txt -output net_youtube_dense.txt -depth 2 -threshold 1000
  ```

* 两次运行line，分别得到一阶相似度和二阶相似度下的embedding结果

  ```shell
  ./line -train net_youtube_dense.txt -output vec_1st_wo_norm.txt -binary 1 -size 128 -order 1 -negative 5 -samples 10000 -threads 40
  ./line -train net_youtube_dense.txt -output vec_2nd_wo_norm.txt -binary 1 -size 128 -order 2 -negative 5 -samples 10000 -threads 40
  ```

* 利用normalize程序将实验结果进行归一化

  ```shell
  ./normalize -input vec_1st_wo_norm.txt -output vec_1st.txt -binary 1
  ./normalize -input vec_2nd_wo_norm.txt -output vec_2nd.txt -binary 1
  ```

* 使用concatenate程序连接一阶嵌入和二阶嵌入的结果

  ```shell
  ./concatenate -input1 vec_1st.txt -input2 vec_2nd.txt -output vec_all.txt -binary 1
  ```

## 编译LINE源码

编译LINE源码时，会遇到一些问题：

[linux下GSL安装](https://blog.csdn.net/waleking/article/details/8265008/)

注意，可能会报错：

```sh
line.cpp:(.text+0x30b8): undefined reference to `gsl_rng_uniform'
```

这时候，需要在编译选项

```sh
-lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result
```

中加入

```sh
-lgsl -lgslcblas
```

就好啦。

具体参见[linux下GSL安装](https://blog.csdn.net/waleking/article/details/8265008/)、[can't link GSL properly?](https://www.daniweb.com/programming/software-development/threads/289812/can-t-link-gsl-properly)。



```sh
../bin/reconstruct -train ../data/net_youtube.txt -output ../data/net_youtube_dense.txt -depth 2 -threshold 1000
```

会出现

```
../bin/reconstruct: error while loading shared libraries: libgsl.so.23: cannot open shared object file: No such file or directory
```

解决办法：

```
export LD_LIBRARY_PATH=/usr/local/lib
```

具体参见[error while loading shared libraries: libgsl.so.23: cannot open shared object file: No such file or directory](https://stackoverflow.com/questions/45665878/a-out-error-while-loading-shared-libraries-libgsl-so-23-cannot-open-shared)。



# 源码解析

在Youtube数据集上运行算法的示例在train_youtube.bat / train_youtube.sh文件中提供。算法运行分为五步：

- 将单向的关系变为双向的关系，因为youtobe好有关系是无向图

  ```shell
  python3 preprocess_youtube.py youtube-links.txt net_youtube.txt
  ```

- 通过reconstruct程序对原网络进行重建（1h）

  ```shell
  ./reconstruct -train net_youtube.txt -output net_youtube_dense.txt -depth 2 -threshold 1000
  ```

- 两次运行line，分别得到一阶相似度和二阶相似度下的embedding结果

  ```shell
  ./line -train net_youtube_dense.txt -output vec_1st_wo_norm.txt -binary 1 -size 128 -order 1 -negative 5 -samples 10000 -threads 40
  ./line -train net_youtube_dense.txt -output vec_2nd_wo_norm.txt -binary 1 -size 128 -order 2 -negative 5 -samples 10000 -threads 40
  ```

- 利用normalize程序将实验结果进行归一化

  ```shell
  ./normalize -input vec_1st_wo_norm.txt -output vec_1st.txt -binary 1
  ./normalize -input vec_2nd_wo_norm.txt -output vec_2nd.txt -binary 1
  ```

- 使用concatenate程序连接一阶嵌入和二阶嵌入的结果

  ```shell
  ./concatenate -input1 vec_1st.txt -input2 vec_2nd.txt -output vec_all.txt -binary 1
  ```

下面我们依次分析这些源码：

## train_youtube.sh

这个代码很简单，就是上述的流程。建议自己先把youtube-links.txt数据下下来，把那段下载的代码屏蔽掉，这样快一些。

```shell
#!/bin/sh

g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result line.cpp -o line -lgsl -lm -lgslcblas
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result reconstruct.cpp -o reconstruct
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result normalize.cpp -o normalize
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result concatenate.cpp -o concatenate

wget http://socialnetworks.mpi-sws.mpg.de/data/youtube-links.txt.gz
gunzip youtube-links.txt.gz

python3 preprocess_youtube.py youtube-links.txt net_youtube.txt
./reconstruct -train net_youtube.txt -output net_youtube_dense.txt -depth 2 -threshold 1000
./line -train net_youtube_dense.txt -output vec_1st_wo_norm.txt -binary 1 -size 128 -order 1 -negative 5 -samples 10000 -threads 40
./line -train net_youtube_dense.txt -output vec_2nd_wo_norm.txt -binary 1 -size 128 -order 2 -negative 5 -samples 10000 -threads 40
./normalize -input vec_1st_wo_norm.txt -output vec_1st.txt -binary 1
./normalize -input vec_2nd_wo_norm.txt -output vec_2nd.txt -binary 1
./concatenate -input1 vec_1st.txt -input2 vec_2nd.txt -output vec_all.txt -binary 1

cd evaluate
./run.sh ../vec_all.txt
python3 score.py result.txt
```

## reconstruct.cpp

这个代码的作用是，对于少于指定的边数的节点，扩展



##line.cpp

 这个程序和word2vec的风格很像，估计就是从word2vec改的。

首先在main函数这，特别说明一个参数：`total_samples`，这个参数是总的训练次数，LINE没有训练轮数的概念，因为LINE是随机按照权重选择边进行训练的。

我们直接看TrainLINE()函数中的TrainLINEThread()这个函数，多线程跑的就是这个函数。

训练结束条件是，当训练的次数超过`total_samples`的次数以后就停止训练。如下：

```c++
if (count > total_samples / num_threads + 2) break;
```

首先要按边的权重采集一条边edge(u, v)，得到其这条边的起始点u和目标点v：

```c++
curedge = SampleAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
u = edge_source_id[curedge];
v = edge_target_id[curedge];
```

然后最核心的部分就是负采样并根据损失函数更新参数：

```c++
lu = u * dim;
for (int c = 0; c != dim; c++) vec_error[c] = 0;

// NEGATIVE SAMPLING
for (int d = 0; d != num_negative + 1; d++)
{
    if (d == 0)
    {
        target = v;
        label = 1;
    }
    else
    {
        target = neg_table[Rand(seed)];
        label = 0;
    }
    lv = target * dim;
    if (order == 1) Update(&emb_vertex[lu], &emb_vertex[lv], vec_error, label);
    if (order == 2) Update(&emb_vertex[lu], &emb_context[lv], vec_error, label);
}
for (int c = 0; c != dim; c++) emb_vertex[c + lu] += vec_error[c];
```

很显然，1阶关系训练的是两个节点的`emb_vertex`，而2阶关系训练的是开始节点的`emb_vertex`（节点本身的embedding）和目标节点的`emb_context`（节点上下文的embedding）。

接下来进入最关键的权值更新函数`Update()`：

```c++
/* Update embeddings */
void Update(real *vec_u, real *vec_v, real *vec_error, int label)
{
	real x = 0, g;
	for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
	g = (label - FastSigmoid(x)) * rho;
	for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
	for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
}
```

这时我们需要回到论文中，看公式（8）和公式（7）：
$$
\begin{aligned}
\frac{\partial O_2}{\partial \overrightarrow{u}_i}&=w_{ij}\cdot \frac{\partial\ \text{log}\ p_2(v_j|v_i)}{\partial \overrightarrow{u}_i}\\
&=\frac{\partial\ \text{log}\ p_2(v_j|v_i)}{\partial \overrightarrow{u}_i}\ (w_{ij}\text{通过Alias采样来近似})\\
&=\frac{\partial\ \text{log}\ \sigma({\overrightarrow{u}'_j}^T\cdot \overrightarrow{u}_i)+\sum_{i=1}^KE_{v_n\sim P_n}\left[\text{log}\ \sigma(-{\overrightarrow{u}'_n}^T\cdot \overrightarrow{u}_i)\right]}{\partial \overrightarrow{u}_i}\\
&=\frac{\partial\ \text{log}\ \sigma({\overrightarrow{u}'_j}^T\cdot \overrightarrow{u}_i)+\sum_{i=1}^KE_{v_n\sim P_n}\left[\text{log}\ \left(1-\sigma({\overrightarrow{u}'_n}^T\cdot \overrightarrow{u}_i)\right)\right]}{\partial \overrightarrow{u}_i}\\
&=\frac{\partial\ \text{log}\ \sigma({\overrightarrow{u}'_j}^T\cdot \overrightarrow{u}_i)}{\partial \overrightarrow{u}_i}+\frac{\sum_{i=1}^KE_{v_n\sim P_n}\partial\ \left[\text{log}\ \left(1-\sigma({\overrightarrow{u}'_n}^T\cdot \overrightarrow{u}_i)\right)\right]}{\partial \overrightarrow{u}_i}\\
&=\left[1-\sigma({\overrightarrow{u}'_j}^T\cdot \overrightarrow{u}_i)\right]+\sum_{i=1}^KE_{v_n\sim P_n} \left[0-\sigma({\overrightarrow{u}'_n}^T\cdot \overrightarrow{u}_i)\right]
\end{aligned}
$$
这下代码应该就很容易能理解了。

上式中的
$$
E_{v_n\sim P_n}
$$
对应TrainLINEThread()负采样部分中的

```c++
target = neg_table[Rand(seed)];
label = 0;
```

边eage(u, v)中的v会在每次update时更新，u会在负采样完之后统一更新。

```c++
// 边eage(u, v)中的v会在每次update时更新
void Update(real *vec_u, real *vec_v, real *vec_error, int label)
{
	xxxxx
	for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
	for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
}

void *TrainLINEThread(void *id)
{
	xxx
	while (1)
	{
		// NEGATIVE SAMPLING
		for (int d = 0; d != num_negative + 1; d++)
		{
			xxx
		}
        // 边eage(u, v)中的u会在负采样完之后统一更新
		for (int c = 0; c != dim; c++) emb_vertex[c + lu] += vec_error[c];

		count++;
	}
	xxx
}
```

## 疑问

1、本身对于LINE的疑问：reconstruct的作用其实就是把一阶关系不够的节点用其二阶节点补充。但是LINE训练的时候，一阶二阶并没有区分训练数据。

2、我本身的问题，reconstruct只选取二阶关系的话，那么两种embedding是怎么算的。

## 时间复杂度O(1)的Alias离散采样算法

**Alias-Sampling-Method**

问题：比如一个随机事件包含四种情况，每种情况发生的概率分别为：
$$
\frac{1}{2},\ \frac{1}{3},\ \frac{1}{12},\ \frac{1}{12}
$$
，问怎么用产生符合这个概率的采样方法。

**最容易想到的方法**

我之前有在[【数学】均匀分布生成其他分布的方法](https://blog.csdn.net/haolexiao/article/details/65157026)中写过均匀分布生成其他分布的方法，这种方法就是产生0~1之间的一个随机数，然后看起对应到这个分布的CDF中的哪一个，就是产生的一个采样。比如落在0~1/2之间就是事件A，落在1/2~5/6之间就是事件B，落在5/6~11/12之间就是事件C，落在11/12~1之间就是事件D。 

但是这样的复杂度，如果用BST树来构造上面这个的话，时间复杂度为O(logN)，有没有时间复杂度更低的方法。

**一个Naive的办法**

一个Naive的想法如下： 









# 参考资料

* [Embedding算法Line源码简读](https://blog.csdn.net/daiyongya/article/details/80963767)
* [LINE实验](https://www.jianshu.com/p/f6a9af93d081)

"LINE源码说明与运行"参考此博客。



* [【数学】时间复杂度O(1)的离散采样算法—— Alias method/别名采样方法](https://blog.csdn.net/haolexiao/article/details/65157026)

"时间复杂度O(1)的Alias离散采样算法"参考了此博客。