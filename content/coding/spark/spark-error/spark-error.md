# Saprk任务问题手收集

* [返回顶层目录](../../../SUMMARY.md)
* [返回上层目录](../spark.md)



# Serialized task exceeds max allowed

用sc.parallelize(data,slices)时，如果data数据过大，易出现该问题：

```scala
User class threw exception: java.util.concurrent.ExecutionException:
org.apache.spark.SparkException
: Job aborted due to stage failure: Serialized task 523:49 was 146289487 bytes, which exceeds max allowed: 
spark.rpc.message.maxSize (134217728 bytes).
Consider increasing spark.rpc.message.maxSize 
or using broadcast variables for large values.
```

从上面的异常提示，已经很明显了，就是默认driver向executor上提交一个任务，它的传输数据不能超过128M，如果超过就抛出上面的异常。

方法一：使用广播变量传输

方法二：调大spark.rpc.message.maxSize的值，默认是128M，我们可以根据需要进行适当调整。在使用spark-submit提交任务的时候，加上配置即可：`--conf "spark.rpc.message.maxSize=512" //代表每个分区允许发送的最大值是512M`。

出错的地方：

```scala
sc.parallelize(modelBuffer,30).saveAsTextFile(feature_embedding_path)
```







# 参考资料

* [Spark任务两个小问题笔记](https://my.oschina.net/u/1027043/blog/1595293)

“Serialized task exceeds max allowed”参考了此资料。
