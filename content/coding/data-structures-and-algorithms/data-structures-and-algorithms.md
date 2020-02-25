# 数据结构与算法

* [返回顶层目录](../../SUMMARY.md)
* [数据结构](data-structures/data-structures.md)
* [算法思想](algorithms/algorithms.md)
* [剑指offer](jianzhi-offer/jianzhi-offer.md)
* [leetcode](leetcode/leetcode.md)

![map](pic/map.jpg)





# c++中常见的STL容器类型

## 顺序容器

元素的插入位置和元素的值无关，只跟插入的时机有关。

### vector向量

- assign

  vector有个函数assign, 可以帮助执行赋值操作。**assign会清空你的容器**。

  函数原型：

  ```c++
  void assign(const_iterator first,const_iterator last);
  // vector.assign(que.begin(), que.end());
  void assign(size_type n,const T& x = T());
  // vector.assign(3, 2.1);// 3个2.1
  ```

  功能：将区间[first, last)的元素赋值到当前的vector容器中，或者赋n个值为x的元素到vector容器中，这个容器会清除掉vector容器中以前的内容。

- begin(), end()

  **这两个类似于数组的地址指针**

  begin函数原型:

  ```c++
  iterator begin();
  const_iterator begin();
  ```

  功能：返回一个当前vector容器中起始元素的迭代器。

  end函数原型：

  ```c++
  iterator end();
  const_iterator end();
  ```

  功能：返回一个当前vector容器中末尾元素的迭代器。

  **可以通过使用 \* vector.begin() 或 * vector.end() 来获得 vector 中第一个或最后一个的值；**

  **也可以直接使用 vector.front() 、vector.back() 来得到 vector 首尾的值。**

- front(), back()

  front函数原型：

  ```c++
  reference front();
  const_reference front();
  ```

  功能：返回当前vector容器中起始元素的引用。

  back函数原型：

  ```c++
  reference back();
  const_reference back();
  ```

  功能：返回当前vector容器中末尾元素的引用。

### deque双队列

- push_front, pop_front()
- push_back, push_back()



### list链表

* iterator

  ```c++
  list<int>::iterator current = numbers.begin();
  ```

* erase

  ```c++
  numbers.erase(current);
  ```





## 关联容器

关联容器内的元素是排序的，插入任何元素，都按照相应的排序专责来确定其位置。

关联式容器的特点是在查找时具有非常好的性能。

通常以平衡二叉树方式实现，插入、查找和删除的时间都是O(logN)。

### set集合



### multiset





### map映射



### multimap



## 容器适配器

### stack栈



### queue队列



### priority_queue优先级队列

优先级高的元素先出。





# c++中常见的函数

## 排序sort

sort(first_pointer,first_pointer+n,cmp)

该函数可以给数组，或者链表list、向量排序。

实现原理：sort并不是简单的快速排序，它对普通的快速排序进行了优化，此外，它还结合了插入排序和推排序。系统会根据你的数据形式和数据量自动选择合适的排序方法，这并不是说它每次排序只选择一种方法，它是在一次完整排序中不同的情况选用不同方法，比如给一个数据量较大的数组排序，开始采用快速排序，分段递归，分段之后每一段的数据量达到一个较小值后它就不继续往下递归，而是选择插入排序，如果递归的太深，他会选择推排序。

此函数有3个参数：

* 参数1：第一个参数是数组的首地址，一般写上数组名就可以，因为数组名是一个指针常量。

* 参数2：第二个参数相对较好理解，即首地址加上数组的长度n（代表尾地址的下一地址）。

* 参数3：默认可以不填，如果不填sort会**默认按数组升序排序**。也就是1,2,3,4排序。也可以自定义一个排序函数，改排序方式为降序什么的，也就是4,3,2,1这样。

使用此函数需先包含：

```c++
#include <algorithm>
```

并且导出命名空间：

```c++
using namespace std;
```

## 数值转字符串to_string

功能：将数值转化为字符串。返回对应的字符串。

函数原型：

```c++
string to_string (int val);
string to_string (long val);
string to_string (long long val);
string to_string (unsigned val);
string to_string (unsigned long val);
string to_string (unsigned long long val);
string to_string (float val);
string to_string (double val);
string to_string (long double val);
```

## string类的相关函数

### substr提取部分字符串

```c++
string str_new = str.substr(0, length);
```

### c_str()转为char\*数组

```c++
const char* cstr = str.c_str();
```



# 参考资料

* [C++模版与STL库介绍](https://wenku.baidu.com/view/93f33b3b192e45361066f5eb.html)

“c++中常见的STL容器类型”参考此课件。

===

[学习「数据结构与算法」，一个很赞的网站](https://zhuanlan.zhihu.com/p/31534696?utm_source=wechat_session&utm_medium=social&utm_oi=1107657858870767616)

[剑指Offer系列刷题笔记汇总](https://cuijiahua.com/blog/2018/02/basis_67.html)

[剑指Offer-在线编程-牛客网](https://www.nowcoder.com/ta/coding-interviews?query=&asc=true&order=&page=1)

[牛客网-leetcode在线编程](https://www.nowcoder.com/ta/leetcode)