# LINE: Large-scale Information Network Embedding

* [返回顶层目录](../../../../../SUMMARY.md)
* [返回上层目录](../embedding.md)





论文地址：https://arxiv.org/pdf/1503.03578.pdf

代码地址：https://github.com/tangjianpku/LINE



# LINE源码解读

安装：

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







https://blog.csdn.net/daiyongya/article/details/80963767



# 参考资料

[Embedding算法Line源码简读](https://blog.csdn.net/daiyongya/article/details/80963767)

"LINE源码解读"参考此博客。





