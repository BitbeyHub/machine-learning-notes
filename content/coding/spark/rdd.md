# RDD编程

* [返回顶层目录](../../SUMMARY.md)
* [返回上层目录](spark.md)



# RDD基本转换操作



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

===

[Spark笔记：RDD基本操作（上）](https://www.cnblogs.com/sharpxiajun/p/5506822.html)