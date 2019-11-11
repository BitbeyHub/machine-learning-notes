# RDD编程

* [返回顶层目录](../../SUMMARY.md)
* [返回上层目录](spark.md)



# RDD基本转换操作

## 输入

### parallelize

调用SparkContext 的 parallelize()，将一个存在的集合，变成一个RDD，这种方式试用于学习spark和做一些spark的测试

> def parallelize[T](seq: Seq[T], numSlices: Int = defaultParallelism)(implicit arg0: ClassTag[T]): RDD[T]
>
> - 第一个参数一是一个 Seq集合
> - 第二个参数是分区数
> - 返回的是RDD[T]

```scala
scala> sc.parallelize(List("shenzhen", "is a beautiful city"))
res1: org.apache.spark.rdd.RDD[String] = ParallelCollectionRDD[1] at parallelize at <console>:22
```



### makeRDD

只有scala版本的才有makeRDD

> def makeRDD\[T\](seq : scala.Seq[T], numSlices : scala.Int = { /* compiled code */ })

跟parallelize类似

```scala
sc.makeRDD(List("shenzhen", "is a beautiful city"))
```



### textFile

调用SparkContext.textFile()方法，从外部存储中读取数据来创建 RDD
例如在我本地F:\dataexample\wordcount\input下有个sample.txt文件，文件随便写了点内容，我需要将里面的内容读取出来创建RDD

```scala
var lines = sc.textFile("F:\\dataexample\\wordcount\\input") 
```

注: textFile支持分区，支持模式匹配，例如把F:\dataexample\wordcount\目录下inp开头的给转换成RDD

```scala
var lines = sc.textFile("F:\\dataexample\\wordcount\\inp*")
```

多个路径可以使用逗号分隔，例如

```scala
var lines = sc.textFile("dir1,dir2",3)
```



## 分区

### coalesce

def coalesce(numPartitions: Int, shuffle: Boolean = false)(implicit ord: Ordering[T] = null): RDD[T]

该函数用于将RDD进行重分区，使用HashPartitioner。

第一个参数为重分区的数目，第二个为是否进行shuffle，默认为false;

以下面的例子来看：

```scala
scala> var data = sc.textFile("/tmp/lxw1234/1.txt")
data: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[53] at textFile at :21
 
scala> data.collect
res37: Array[String] = Array(hello world, hello spark, hello hive, hi spark)
 
scala> data.partitions.size
res38: Int = 2  //RDD data默认有两个分区
 
scala> var rdd1 = data.coalesce(1)
rdd1: org.apache.spark.rdd.RDD[String] = CoalescedRDD[2] at coalesce at :23
 
scala> rdd1.partitions.size
res1: Int = 1   //rdd1的分区数为1
 
 
scala> var rdd1 = data.coalesce(4)
rdd1: org.apache.spark.rdd.RDD[String] = CoalescedRDD[3] at coalesce at :23
 
scala> rdd1.partitions.size
res2: Int = 2   //如果重分区的数目大于原来的分区数，那么必须指定shuffle参数为true，//否则，分区数不便
 
scala> var rdd1 = data.coalesce(4,true)
rdd1: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[7] at coalesce at :23
 
scala> rdd1.partitions.size
res3: Int = 4
```

### repartition

def repartition(numPartitions: Int)(implicit ord: Ordering[T] = null): RDD[T]

该函数其实就是coalesce函数第二个参数为true的实现

```scala
scala> var rdd2 = data.repartition(1)
rdd2: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[11] at repartition at :23
 
scala> rdd2.partitions.size
res4: Int = 1
 
scala> var rdd2 = data.repartition(4)
rdd2: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[15] at repartition at :23
 
scala> rdd2.partitions.size
res5: Int = 4
```



# 参考资料

* [Spark算子：RDD基本转换操作(2)–coalesce、repartition](http://lxw1234.com/archives/2015/07/341.htm)

"分区"参考此文章。

* [spark RDD算子（一） parallelize，makeRDD，textFile](https://blog.csdn.net/T1DMzks/article/details/70189509)

"输入"参考此文章。

===

[Spark笔记：RDD基本操作（上）](https://www.cnblogs.com/sharpxiajun/p/5506822.html)