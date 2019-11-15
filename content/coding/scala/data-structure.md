# 数据结构

* [返回顶层目录](../../SUMMARY.md)
* [返回上层目录](scala.md)





# 其他

## 使用Range来填充一个集合

Problem
​    你想要使用Range来填充一个List，Array，Vector或者其他的sequence。

Solution
​    对于支持range方法的集合你可以直接调用range方法，或者创建一个Range对象然后把它转化为一个目标集合。

在第一个解决方案中，我们调用了伴生类的range方法，比如Array，List，Vector，ArrayBuffer等等：

```scala
scala> Array.range(1, 10)
res83: Array[Int] = Array(1, 2, 3, 4, 5, 6, 7, 8, 9)
 
scala> List.range(1, 10)
res84: List[Int] = List(1, 2, 3, 4, 5, 6, 7, 8, 9)
 
scala> Vector.range(0, 10, 2)
res85: scala.collection.immutable.Vector[Int] = Vector(0, 2, 4, 6, 8)
```

对于一些集合，比如List，Array，你也可以创建一个Range对象，然后把它转化为相应的目标集合：

```scala
scala> val a = (1 to 10).toArray
a: Array[Int] = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
 
scala> val l = (1 to 10) by 2 toList
warning: there were 1 feature warning(s); re-run with -feature for details
l: List[Int] = List(1, 3, 5, 7, 9)
 
scala> val l = (1 to 10).by(2).toList
l: List[Int] = List(1, 3, 5, 7, 9)
```

 我们来看看那些集合可以由Range直接转化的：

```scala

def toArray: Array[A]
def toBuffer[A1 >: Int]: Buffer[A1]
def toIndexedSeq: IndexedSeq[Int]
def toIterator: Iterator[Int]
def toList: scala.List[Int]
def toMap[T, U]: collection.Map[T, U]
def toParArray: ParArray[Int]
def toSet[B >: Int]: Set[B]
def toStream: Stream[Int]
def toTraversable: collection.Traversable[Int]
def toVector: scala.Vector[Int]
```

使用这种方案我们可以把Range转为Set等，不支持range方法的集合类：

```scala
scala> val set = Set.range(0, 5)
<console>:8: error: value range is not a member of object scala.collection.immutable.Set
       val set = Set.range(0, 5)
                     ^
 
scala> val set = Range(0, 5).toSet
set: scala.collection.immutable.Set[Int] = Set(0, 1, 2, 3, 4)
 
scala> val set = (0 to 10 by 2).toSet
set: scala.collection.immutable.Set[Int] = Set(0, 10, 6, 2, 8, 4)
```

 你也可以创建一个字符序列：

```scala
scala> val letters = ('a' to 'f').toList
letters: List[Char] = List(a, b, c, d, e, f)
 
scala> val letters = ('a' to 'f' by 2).toList
letters: List[Char] = List(a, c, e)
```

Range还能用于for循环：

```scala
scala> for(i <- 0 until 10 by 2) println(i)
0
2
4
6
8
```

**Discussion**

通过对Range使用map方法，你可以创建出了Int，char之外，其他元素类型的集合

```scala
scala> val l = (1 to 3).map(_ * 2.0).toList
l: List[Double] = List(2.0, 4.0, 6.0)
```

使用同样的方案，你可以创建二元祖集合：

```scala
scala> val t = (1 to 5).map(e => (e, e*2))
t: scala.collection.immutable.IndexedSeq[(Int, Int)] = Vector((1,2), (2,4), (3,6), (4,8), (5,10))
```

二元祖集合很容易转换为Map：

```scala
scala> val map = t.toMap
map: scala.collection.immutable.Map[Int,Int] = Map(5 -> 10, 1 -> 2, 2 -> 4, 3 -> 6, 4 -> 8)
```





# 参考资料

* [scala使用Range来填充一个集合](https://blog.csdn.net/qq_36330643/article/details/76483551)

"使用Range来填充一个集合"参考此博客。



