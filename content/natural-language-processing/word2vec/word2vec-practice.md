# word2vector实践

- [返回顶层目录](../../../SUMMARY.md)
- [返回上层目录](word2vec.md)



# 原版word2vec实践

官方使用说明：[word2vec](https://code.google.com/archive/p/word2vec/)，这个并没有实践过，只是放在这里，下面是经过实践检验的：

下面的介绍仅仅适用于linux系统，windows其实也差不多，只是不能用sh语言了，需要自己去手动编译和运行程序或者自己写bat语言。

## 下载源码

github:[dav/word2vec](https://github.com/dav/word2vec)

## 下载训练数据

下载训练数据：[text8.zip](http://mattmahoney.net/dc/text8.zip)，用`unzip text8.zip`将其解压缩到`\data`文件夹中，文件名为text8，这个解压后的文件text8就是模型训练需要的数据了。

其实在sh文件中会自动下载text8，但是还是自己先下载下来吧。

## 开始训练模型

（1）由c源码生成可执行文件

两种方法：

第一种方法：进入word2vec的根目录，然后命令行运行`make build`把`\src`文件夹中的c文件编译为可执行文件放到`\bin`文件夹中。

第二种方法：进入word2vec\src目录，然后命令行运行`make`把`\src`文件夹中的c文件编译为可执行文件放到`\bin`文件夹中。

（2）运行可执行文件开始训练模型

进入根目录下的`\scripts`文件夹，然后运行sh文件demo-classes.sh：`sh demo-classes.sh`，等待程序运行完毕（可能需要五分钟到一个小时之间，看机器性能）。

## 得到结果文件

等训练结束后，会在`\data`文件夹中生成两个文件：classes.txt和classes.sorted.txt，其中classes.txt就是我们需要的文件。打开就能看到每个单词训练的embedding向量了。

## 查看TopN相似度

进入根目录下的`\scripts`文件夹，然后运行sh文件demo-word.sh：`sh demo-word.sh`，然后按照提示输入单词，查看与其相近的前40个词及相似程度。

相似程度的代码是distance.c，很简单，有个博客有对这个代码的注释：[Word2Vec代码注解-distance](https://blog.csdn.net/a785143175/article/details/23771625)

# 参考资料

* [使用word2vec（C语言版本）训练中文语料 并且将得到的vector.bin文件转换成txt文件](https://blog.csdn.net/zwwhsxq/article/details/77200129)
* [Deep Learning 实战之 word2vec](https://kexue.fm/usr/uploads/2017/04/146269300.pdf)
* [word2vec 入门教程](https://blog.csdn.net/bitcarmanlee/article/details/51182420)
* [word2vec-google code](https://code.google.com/archive/p/word2vec/)


