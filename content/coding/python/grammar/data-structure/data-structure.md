# 数据结构

* [返回顶层目录](../../../SUMMARY.md#目录)
* [返回上层目录](../python.md)
* [range()](#range())
* [字符串](#字符串)
  * [字符串操作](#字符串操作)
  * [字符串常用内置方法](#字符串常用内置方法)
* [列表](#列表)
  * [对列表进行增删改查](#对列表进行增删改查)
  * [列表去重](#列表去重)
* [元组](#元组)
* [字典](#字典)
  * [基本操作](#基本操作)
  * [字典的增删改查操作](#字典的增删改查操作)
  * [使用循环来遍历字典](#使用循环来遍历字典)
  * [清空字典](#清空字典)
* [集合](#集合)



# range()

range(开始，结束)，返回一个序列，**左闭右开**

```python
for i in range(10): #等价于range(0,10)
	print(i)
```

range(开始，结束，步长)

```python
# 打印0到10的偶数
for i in range(0,11,2):
	print(i)
```

# 字符串

## 字符串操作

- 字符串变量定义

  s = "hello"

  s = 'hello'

- 组成字符串的方式

  - 使用“+”将两个字符串连接成一个新的字符串
  - 组成字符串格式化符号

- 下标hello[01234]

  - 通过下标获取指定位置的字符，string_name[index]

- 切片

  - 切片的语法：string_name[起始:结束:步长]

代码

- 基本操作

```python
s = "hello"
# 通过交表获取指定位置元素
print(s[1])
# 获取字符串长度
print(len(s))
# 循环遍历字符串
for i in range(0,len(s)):
    print(s[i],end="")
print("")
# 脚标 用负数表示倒着数
print(s[-1])# 倒数第一个
print(s[-2])# 倒数第二个
```

**切片**

- 基本操作

```python
# 切片
# 注意：切片切出来的字符串是左闭右开的
line = "zhangsan,20"
name = line[0:8:1]
print(name)
age = line[9::1]# 截取到最后，可以不加结束序号
print(age)
```

- 切片步长

```python
# 切片步长
s = "abcde"
print(s[1:])# bcde
print(s[1:-1])# bcd
print(s[1:-2])# bc
# 隔一个位置取一个元素
print(s[0::2])# 等价于
print(s[::2])# ace
```

## 字符串常用内置方法

- find

  在字符串中查找指定的子字符串是否存在，如果存在则返回第一个子字符串的起始下标，如果不存在则返回-1

- count

  在字符串中统计包含的子字符串的个数

- replace

  使用新的子字符串替换指定的子字符串，返回新的字符串

- split

  按照指定的分隔符字符串，返回分割之后的所有元素的列表

```python
# find
line = "hello world hello python"
print(line.find("hello"))# 第一个子字符串的起始脚标
print(line.find("hello", 6))# 从第六个脚标其开始查找 12
print(line.find("java")) # 不存在，返回-1
# count
print(line.count("world")) # 出现的次数
# replace 字符串是不可变类型
new_line = line.replace("hello", "qqq")
print(new_line)

# split分割
line_list = line.split(" ")
print(line_list)
```

- startswith

  判断字符串是否以指定前缀开头，返回值为True或False

- endswith

  判断字符串是否以指定后缀结束，返回值为True或False

```python
# startswith
files = ["20171201.txt","20180101.log"]
for item in files:
    if item.startswith("2018") and item.endswith("log"):
        print("2018年待处理日志：{}".format(item))
```

- upper

  字符串所有字符大写

- lower

  字符串所有字符小写

```python
# upper lower 大小写
content = input("是否继续，继续输入yes，退出输入no")
if content.lower() == "yes":
    print("欢迎继续使用")
else:
    print("退出，请取卡")
```

# 列表

可理解为柜子，柜子里有不同的抽屉，可存储不同类型的值。

- 可存储相同或者不同类型数据的集合

- 列表定义

  - name_list = ["zhangsan","lisi","wangwu"]

- 顺序存储，可通过下标获取内部元素

  name_list[0]

  name_list[1]

- 内容可变，可通过下角标修改元素值

  name_list[0] = "xiaobai"

- 使用循环遍历列表

- 嵌套列表

代码

基本操作

```python
name_list = ["zhangsan", "lisi", "wangwu"]
print(name_list)# ['zhangsan', 'lisi', 'wangwu']
print(type(name_list))# 类型<class 'list'>
# 脚标获取列表元素
print(name_list[0])
```

存储不同类型的元素，遍历列表

```python
# 存储不同类型的数据
info_list = ["zhangsan", 20, 180.5, 80, True]
print(info_list[4])# True
info_list[3] = 70
print(info_list)# ['zhangsan', 20, 180.5, 70, True]

# 遍历列表，获取列表所有元素
# while循环
i = 0
while i < len(info_list):
    print(info_list[i])
    i += 1
# for循环，通过脚标
for i in range(len(info_list)):
    print(info_list[i])
# for循环，通过序列的每一个元素
for item in info_list:
    print(item)
```

嵌套列表

```python
# 嵌套列表
info_lists = [["zhangsan", 20], ["lisi", 30], ["wangwu", 40]]
print(info_lists[0])# ['zhangsan', 20]
print(info_lists[0][0])# zhangsan

# 循环遍历嵌套列表
for person in info_lists:
    for item in person:
        print(item)
    print("-----------")
```

## 对列表进行增删改查

- append()/insert()添加元素

  - append()向列表末尾添加元素

    ```python
    # append向列表末尾添加元素
    info_lists = [["zhangsan", 20], ["lisi", 30], ["wangwu", 40]]
    info_lists.append(["xiaobai", 25])
    print(info_lists)
    # [['zhangsan', 20], ['lisi', 30], ['wangwu', 40], ['xiaobai', 25]]
    ```

  - insert()可指定位置添加元素

    ```python
    # insert(index, item)
    info_lists = [["zhangsan", 20], ["lisi", 30], ["wangwu", 40]]
    info_lists.insert(1,["wangmazi", 23])
    print(info_lists)
    # [['zhangsan', 20], ['wangmazi', 23], ['lisi', 30], ['wangwu', 40]]
    ```

- “+”组合两个列表生成新的列表

  ```python
  # 两个列表元素组合生成新的列表
  name_list1 = ["唐僧","悟空","八戒"]
  name_list2 = ["沙僧", "白龙马"]
  new_list = name_list1 + name_list2
  print(new_list)# ['唐僧', '悟空', '八戒', '沙僧', '白龙马']
  ```

- extend()向调用它的列表中添加另外一个列表的元素

  ```python
  name_list1 = ["唐僧","悟空","八戒"]
  name_list2 = ["沙僧", "白龙马"]
  name_list1.extend(name_list2)
  print(name_list1)# ['唐僧', '悟空', '八戒', '沙僧', '白龙马']
  ```


- del()/pop()/remove()删除元素

  ```python
  # 删除
  name_list = ["唐僧","悟空","八戒", "沙僧", "白龙马"]
  # 指定脚标
  del name_list[1]
  print(name_list)
  # 指定元素名称
  name_list.remove("八戒")
  print(name_list)
  # 删除最后一个元素(未指定删除位置)
  name_list.pop()
  print(name_list)
  # 删除某个脚标
  name_list.pop(1)
  print(name_list)
  ```

- 切片

  - 和字符串的切片操作相同

    ```python
    name_list = ["唐僧","悟空","八戒", "沙僧", "白龙马"]
    print(name_list[1::2])# ['悟空', '沙僧']
    ```

- in/not in 判断元素在列表中是否存在

  ```python
  name_list = ["唐僧","悟空","八戒", "沙僧", "白龙马"]
  print("悟空" in name_list)# True
  print("如来" in name_list)# False
  print("悟空" not in name_list)# False
  print("如来" not in name_list)# True
  ```

- sort()列表内元素重排序

  - 默认从小到大排列

    ```python
    # sort()
    num_list = [6, 3, 12, 1]
    num_list.sort() # 默认升序
    print(num_list)# [1, 3, 6, 12]
    num_list.sort(reverse=True) # 倒序排列
    print(num_list)# [12, 6, 3, 1]
    ```

- reverse()列表内容倒置

  ```python
  num_list = [6, 3, 12, 1]
  num_list.reverse()
  print(num_list)#[1, 12, 3, 6]
  ```

- count()统计列表内指定元素个数

  ```python
  num_list = [6, 3, 12, 1, 6, 6]
  print(num_list.count(6))# 3
  ```

## 列表去重

使用内置set方法来去重：

```python
lst1 = [2, 1, 3, 4, 1]
lst2 = list(set(lst1))
print(lst2)
# [1, 2, 3, 4]
```



# 元组

- 循序存储相同/不同类型的元素

- 元组定义，使用()将元素括起来，元素之间用","隔开

- 特性：**不可变，不支持增删改查**

- 查询：通过下标查询元组指定位置的元素

- 空元组的定义：none_tuple = ()

- 只包含一个元素的元组：one_tuple = ("one",)

  **定义一个元素的元组，括号里一定要有一个逗号(,)**

- 循环遍历元组



列表用于存储可变的元素，一般存储相同类型的元素。

元组不可变，通常存储一些不希望被改变的信息，如用户名，密码等。

```python
#定义元组，存储数据库信息
db_info = ("192.169.1.1", "root", "root123")
# 通过脚标查询元组中的元素
ip = db_info[0]
port = db_info[1]
print("ip:{},port:{}".format(ip, port))
# ip:192.169.1.1,port:root
print(type(ip))#<class 'str'>

# 通过脚标来修改元组指定元素的值，这是不行的
db_info[1] = 8080
print(db_info)
# TypeError: 'tuple' object does not support item assignment

# del删除元组指定位置的元素，这是不行的
del db_info[1]
# TypeError: 'tuple' object doesn't support item deletion

# 定义一个元组
one_tuple = (123,)
print(one_tuple)# (123,)
print(type(one_tuple))# <class 'tuple'>
# 错误的定义只包含一个元素的元组，少了元素后的逗号。
one_tuple1 = (123)
print(one_tuple1)# 123
print(type(one_tuple1))# <class 'int'>

# 定义空元组
none_tuple = ()
print(none_tuple)
print(type(none_tuple))
```

循环遍历元组

```python
# 循环遍历
db_info = ("192.169.1.1", "root", "root123")
# for循环
for item in db_info:
    print(item)

# while循环
i = 0
while i < len(db_info):
    print(db_info[i])
    i += 1
```

# 字典

- 存储Key-Value键值对类型的数据
- 字典定义：{Key1 : value1,  key2 : value2,  ...}
- 查询：根据Key查找Value
- 字典具有添加、修改、删除操作
- 内置方法get、keys、values、items、clear
- 循环遍历字典



## 基本操作

```python
user_if_dict = {"name":"悟空","age":100,"gender":"male","job":"取经"}
print("{}的年龄：{}，性别：{}，工作内容：{}".format(user_if_dict["name"],user_if_dict["age"],user_if_dict["gender"],user_if_dict["job"]))

# 通过Key修改已经存在的值
user_if_dict["job"] = "取经|偷桃"
print(user_if_dict["job"])
```

使用字典的原因：

存储大量数据也能够准确查找和修改。

不支持通过下标来查询指定位置的元素的值。

Key不能重复出现，否则后面的值会覆盖前面的值

```python
user_if_dict = {"name":"悟空","age":100,"gender":"male","job":"取经","name":"白骨精"}
print("{}的年龄：{}，性别：{}，工作内容：{}".format(user_if_dict["name"],user_if_dict["age"],user_if_dict["gender"],user_if_dict["job"]))
```

## 字典的增删改查操作

```python
# 字典的增删改查操作

# 添加一个键值对
user_if_dict = {"name":"悟空","age":100,"gender":"male","job":"取经"}
user_if_dict["tel"] = 13812345678
print(user_if_dict)# 5对
print(len((user_if_dict)))

# 修改字典中的指定的值
user_if_dict["tel"] = 13811118888
print(user_if_dict)
# 删除元素
del user_if_dict["tel"]
print(user_if_dict)

# 查询指定名字的元素
print(user_if_dict["name"])
# 查询不存在的键，会报错
# 解决办法
# 方法一：in or not in
if "tel" in user_if_dict:
    print(user_if_dict["tel"])
else:
    print("\"tel\"不存在")
# 方法二： 字典内置的get方法
# 如果不存在，就会返回一个设定的默认值，用于缺省只补全
print(user_if_dict.get("tel","19911116666"))# None
```

## 使用循环来遍历字典

```python
# 使用循环来遍历字典
# 字典内置的Keys方法，返回所有的Key组成一个序列
user_if_dict = {"name":"悟空","age":100,"gender":"male","job":"取经","name":"白骨精"}
for key in user_if_dict.keys():
    print("{}:{}".format(key, user_if_dict[key]),end="|")
# 字典内置的Values方法，返回所有的Value组成的一个序列
for value in user_if_dict.values():
    print(value)# 只能遍历出字典所有的值
# 返回字典的键值对，组成元组返回
for item in user_if_dict.items():
    print(type(item))# <class 'tuple'>
    print(item)# ('name', '白骨精')
    print(item[0])# Key: name
    print(item[1])# Value: 白骨精
# 用两个变量分别接受字典的Key和Value
for key,value in user_if_dict.items():
    print("{}:{}".format(key,value))
```

## 清空字典

```python
user_if_dict = {"name":"悟空","age":100,"gender":"male","job":"取经","name":"白骨精"}
print(user_if_dict)
user_if_dict.clear()
print(user_if_dict)# {}
```

# 集合

- 无序存储不同数据类型不重复元素的序列

  即使填入多个相同的元素，也会被去重

- 集合定义：name_set={"xiaoming",  "xiaoqiang", "xiaobai"}

- 使set对序列中元素去重，同时创建集合

  例如：name_set = set(["xiaoming", "zhangsan"])

- 创建空集合：none_set = set()

- 使用in和not in判断一个元素在集合中是否存在

- 使用add(元素)方法添加一个元素到集合中

- 使用update(序列)方法将一个序列中的元素添加到集合中，同时对元素去重

- remove(元素)根据元素值删除集合中指定元素，如果元素不存在，则报错

- discard(元素)根据元素值删除集合中指定元素，如果元素不存在，不会引发错误

- pop()随机删除集合中的某个元素，并且返回被删除的元素

- clear()清空集合

- 集合操作

  - 交集intersection(&)
  - 并集union(|)
  - 差集difference(-)
  - 对称差集(^)

基本用法

```python
# 集合的定义，元素去重
student_set = {"zhangsan","lisi","wangwu","zhangsan"}
print(student_set)# 无序，去重
print(len(student_set))
print(type(student_set))
# set(序列)
# set(集合) 对list中的元素去重，并创建一个新的集合
id_list = ["id1", "id2", "id3", "id1", "id2"]
new_set = set(id_list)
print(id_list)
print(new_set)
# set(元组) 对突破了中的元素去重，并创建一个新的集合
id_tuple = ("id1", "id2", "id3", "id1", "id2")
new_set = set(id_tuple)
print(id_tuple)
print(new_set)
```

对于字符串，会打乱元素顺序，并对字符去重。

```python
string_set = set("hello")
print(string_set)
# {'o', 'e', 'l', 'h'}
```

创建空集合

```python
# 创建空集合
none_set = set()
print(none_set)# set()
# 注意，床架空字典是{}
none_dict = {}
print(none_dict)# {}
```

判断存在与否

```python
# in or not in
id_list = ["id1", "id2", "id3", "id1", "id2"]
new_set = set(id_list)
user_id = "id1"
if user_id in new_set:
    print("{}存在".format(user_id))
elif user_id not in new_set:
    print("{}不存在".format(user_id))
```

update添加序列，而add()只能添加一个元素

```python
# update(序列) 重复元素会去重
name_set = {"zhangsan", "lisi"}
# 添加列表元素到集合
name_set.update(["悟空", "八戒"])
print(name_set)
# 添加元组元素到集合
name_set.update(("悟空", "八戒"))
print(name_set)
# 添加多个序列元素到集合
name_set.update(["悟空", "八戒"],["沙僧", "八戒"])
print(name_set)
# 把一个集合并入另一个集合
name_set.update({"张飞","李逵"})
print(name_set)
# add()只能添加一个元素
name_set.add("如来佛")
print(name_set)
```

三种删除操作

```python
# 三种删除操作

#remove(元素)
name_set = {"zhangsan", "lisi", "wangwu"}
name_set.remove("zhangsan")
print(name_set)
# remove删除不存在的元素会报错
# name_set.remove("zhangsan")# KeyError: 'zhangsan'

# discard(元素)
name_set = {"zhangsan", "lisi", "wangwu"}
name_set.discard("zhangsan")
print(name_set)
name_set.discard("zhangsan")

# pop随机删除
name_set = {"zhangsan", "lisi", "wangwu"}
name_set.pop()
print(name_set)
```

交集，并集，差集

```python
# 交集，并集，
# 交集
num_set1 = {1,2,4,7}
num_set2 = {2,5,8,9}
inter_set1 = num_set1 & num_set2
inter_set2 = num_set1.intersection(num_set2)
print(inter_set1)
print(inter_set2)

# 并集
num_set1 = {1,2,4,7}
num_set2 = {2,5,8,9}
union_set1 = num_set1 | num_set2
union_set2 = num_set1.union(num_set2)
print(union_set1)
print(union_set2)

# 差集
num_set1 = {1,2,4,7}
num_set2 = {2,5,8,9}
diff_set1 = num_set1 - num_set2
diff_set2 = num_set1.difference(num_set2)
print(diff_set1)
print(diff_set2)

# 对称差集, 互差再并
num_set1 = {1,2,4,7}
num_set2 = {2,5,8,9}
sym_diff_set1 = num_set1 ^ num_set2
sym_diff_set2 = num_set1.symmetric_difference(num_set2)
print(sym_diff_set1)
print(sym_diff_set2)
```



# 知识点

## range函数的作用

python内置range()函数的作用是什么？它能返回一系列连续增加的整数，它的工作方式类似于分片，可以生成一个列表对象。range函数大多数时常出现在**for循环中**，在for循环中可做为索引使用。其实它也可以出现在任何需要整数列表的环境中，在python 3.0中range函数是一个迭代器，不能print(range(4))

Python3.x 中 range() 函数返回的结果是一个**整数序列的对象**，而**不是列表**。

```python
print(type(range(10)))
#<class 'range'>
```

当你 help(range) 时会看到：

```python
help(range)
#Return an object that produces a sequence of integers from start (inclusive)
#to stop (exclusive) by step.
```

所以，不是列表，但是可以利用 list 函数返回列表，即：

```python
print(list(range(5)))
#[0, 1, 2, 3, 4]
```

## Python 序列的概念及基本操作方法

序列sequence是python中最基本的数据结构，本文先对序列做一个简单的概括，之后简单讲解下所有序列都能通用的操作方法。

**1序列概念**
列表和元组这两种数据类型是最常被用到的序列，python内建序列有六种，除了刚刚有说过的两种类型之外，还有字符串、Unicode字符串、buffer对像和最后一种xrange对像，这几种都是不常使用的。

**2序列通用操作方法**
所有序列类型有一些可以通用的方法，比如：索引、分片、乘、加，检查成员资格。当然还有一些很实用的内建函数，像是计算序列长度，找出序列中的最大或最小无素等。下来就来一一的介绍下序列的基本操作方法吧。

**2.1什么是索引**：序列中的每一个元素都有自己的位置编号，可以通过偏移量索引来读取数据。最开始的第一个元素，索引为0，第二个元素，索引为1，以此类推；也可以从最后一个元素开始计数，最后一个元素的索引是-1，倒数第二个元素的索引就是-2，以此类推。

**2.2序列相加**：相同数据类型序列之间可以相加，不同数据类型序列不能相加。

列表类型序列相加

```python
print([1,2]+[3,4])
#[1, 2, 3, 4]

list1 = list([1,2])
list2 = list([3,4])
list3 = list1 + list2
print(list3)
#[1, 2, 3, 4]
```

字符串类型序列相加

```python
print('hello'+'.python')
#hello.python
```

元组类型序列相加

```python
print((1,2,3)+(4,5,6))
#(1, 2, 3, 4, 5, 6)

a = (1,2,3)
b = (4,5,6)
c = a + b
print(c)
#(1, 2, 3, 4, 5, 6)
```

**2.3序列乘法**：把原序列乘X次，并生成一个新的序列

```python
list1 = list([1,2])
list1_3 = list1 * 3
print(list1_3)
# [1, 2, 1, 2, 1, 2]

list1 = [1,2]
list1_3 = list1 * 3
print(list1_3)
# [1, 2, 1, 2, 1, 2]

list1_3 = [1,2] * 3
print(list1_3)
# [1, 2, 1, 2, 1, 2]
```

**2.4成员资格**：检查某个指定的值是否在序列中，用in布尔运算符来检查，其返回值为True/False。True为真，在这里可以理解为要查找的值在序列中，False结果与其相反。

```python
a = 'iplaypython'
print('i' in a)
print('z' in a)
#True
#False
```

**2.5序列内建函数**：len()函数计算序列内元素数量；min()函数、max()函数分别查找并返回序列中的最大或最小元素。

```python
num = [99,1,55]
print(len(num))
print(min(num))
print(max(num))
#3
#1
#99
```

在操作一组数据时，序列是很好用的数据结构。列表、元组和字符串这几种数据类型是比较常接触到的序列。除了以上讲的序列基本操作方法之外，还有一个比较重要的序列迭代没有讲，这部分内容会单独做讲解。

### 序列迭代





## 赋值、浅拷贝、深拷贝的区别

------

**一、概念**

对于一个对象/结构体

```c++
struct X
{
  int x;//理解为：文件
  int y;//理解为：文件
  int* p;//理解为：文件夹，或者文件的快捷方式
};
```

**赋值(在python中赋值相当于引用)**是指源对象与拷贝对象共用一份实体，仅仅是引用的变量不同（名称不同）。对其中任何一个对象的改动都会影响另外一个对象。也就是说，对于python的直接赋值，X a = {...}，X b = a，这就相当于C++中的引用，b只不过是a的另外一个名字罢了，那自然是其中一个改变必然引起另一个的变化了。

引用就像是复制了文件夹的快捷方式然后重命名。

**浅拷贝**(影子克隆):只复制对象的基本类型,而对象类型仍属于原来的引用，即对于基本类型就不保存内存地址，而直接就是复制数值本身了，对于对象类型，则仍是引用，会相互影响。浅拷贝是指将对象中的数值类型的字段拷贝到新的对象中，而对象中的引用型字段则指复制它的一个引用到目标对象。如果改变目标对象中引用型字段的值他将反映在原始对象中，也就是说原始对象中对应的字段也会发生变化。也就是说，浅拷贝对于所复制的对象什么都不管，看到什么就复制什么，看到int x, int y就复制，看到int* p就复制，那int* p其实是一个指针啊，只相当于钥匙，但是箱子里的东西没有复制啊，所以，X a = {...}，X b = copy.copy(a)，a或b的int x, int b的值变了，另一个值是不会发生变化的，而int* p所指向的值变了，那么另外一个int* p所指向的值必然发生变化，因为都是同一把钥匙啊，开的都是同一个箱子。

浅拷贝就像是，对对象的基本类型则是将原始值复制到开辟的新的内存空间中。对对象的对象类型，则是只复制地址而已。形象的说，就是对文件直接复制，对文件夹仅仅复制快捷方式。 浅拷贝一言以蔽就是存的什么就拷贝什么，这本来就是编译器做的事，管你是不是指针还是什么，文件夹其实你存的是文件夹的地址，而文件夹的内容是存在某处空间的。

**深拷贝**：而对于深拷贝，这一个勤奋的人，他不会只做表面，他会把每一个细节都照顾好。于是，当他遇到指针的时候，他会知道new出来一块新的内存，然后把原来指针指向的值拿过来，这样才是真正的完成了克隆体和原来的物体的完美分离，如果物体比作人的话，那么原来的人的每一根毛细血管都被完美的拷贝了过来，而绝非只是表面。所以，这样的代价会比浅拷贝耗费的精力更大，付出的努力更多，但是是值得的。当原来的物体销毁后，克隆体也可以活的很好。

深拷贝实现代码：

```c++
int *a = new int(42);
int *b = new int(*a);
```

------

**二、深拷贝和浅拷贝的区别**

深拷贝和浅拷贝其实真正的区别在于是否还需申请新的内存空间。  成员不包括指针和引用时，这两种拷贝没区别。浅拷贝是只拷贝指针地址，意思是浅拷贝指针都指向同一个内存空间，当原指针地址所指空间被释放，那么浅拷贝的指针全部失效。对于字符串类型，浅复制是对值的复制，对于对象来说，**浅复制是对对象地址的复制**，并没   有开辟新的栈，也就是复制的结果是两个对象指向同一个地址，修改其中一个对象的属性，则另一个对象的属性也会改变，**而深复制则是开辟新的栈**，两个对象对应两个不同的地址，修改一个对象的属性，不会改变另一个对象的属性。**浅拷贝只是对指针的拷贝，拷贝后两个指针指向同一个内存空间，深拷贝不但对指针进行拷贝，而且对指针指向的内容进行拷贝，经深拷贝后的指针是指向两个不同地址的指针。**

打个比方：我有一个箱子，箱子里只装了一张纸条，纸条上写着保险柜的密码。这时我拿来了第二个箱子。我打开旧箱子，把密码纸上的内容复印了一份，放到新箱子里。这是浅拷贝。我打开旧箱子，取出密码纸，找到保险柜，取出绝密资料，然后我去买了一个一模一样的保险柜，复制了一份新的绝密资料防放到新保险柜里。然后我给新的保险柜创建了一个新密码，放到新箱子里。这是深拷贝。 

再打个比方：你想复制一间房子。深拷贝指的是你重新买了间房，里面的布置也全和原来那间做的一模一样。这时候你有两间一模一样的房子。潜拷贝指的是你想复制一间房，就去重新配了把钥匙。你最终还是只有一间房。只是多了个钥匙而已。

忘掉这两个概念，好好理解什么是指针，什么是内存，内存怎么分布。只要理解了指针之间的相互赋值、传递的行为是怎样的，就自然理解了深拷贝浅拷贝是怎么个情况。并不用特意去记着两个名词。有天赋的一看到指针之间互相赋值只是复制了地址，就能想到某些时候我想要的行为可能不是这样的，那该怎么办。

我吐个槽。。这玩意儿根本就是胡乱扯出来的概念，不是说不好，只是把很自然的事情搞一个高大上的名词，感觉故弄玄虚。说白了，就是类值和类指针行为的区别。对于含有指针成员的类，直接拷贝可能会出现两个对象的指针成员指向同一个数据区。这时候一般先new个内存，然后复制内容 。当然这也得看情况，如果整个类只需要一份这样的数据，就没必要new内存了，直接编译器默认构造函数就行。总结：一般对于堆上的内存才需要深拷贝 。

为了防止内存泄漏，如何在C++中构造一个深拷贝的类的成员函数呢？请看[浅拷贝和深拷贝的区别？](http://www.cnblogs.com/always-chang/p/6107437.html)。

------

**三、用python代码来解释**

**python中的浅拷贝和深拷贝函数**

在python中，对象赋值实际上是对象的引用。当创建一个对象，然后把它赋给另一个变量的时候，python并没有拷贝这个对象，而只是拷贝了这个对象的引用。以下分两个思路来分别理解浅拷贝和深拷贝：

- **浅拷贝**：利用切片操作、工厂方法list方法拷贝、使用copy模块中的copy()函数。
- **深拷贝**：利用copy中的deepcopy方法进行拷贝。

**例子1**

1. **利用切片操作和工厂方法list方法浅拷贝**

```python
jack = ['jack', ['age',20]]
tom = jack[:]
anny = list(jack)

print(id(jack), id(tom), id(anny))# 浅拷贝的对象名称不同
# 204155592 204156296 200960776
# 从id值来看，三者是不同的对象。为tom和anny重新命名为各自的名称：
tom[0] = 'tom'
anny[0] = 'anny'
print(jack, tom, anny)
# ['jack', ['age', 20]] ['tom', ['age', 20]] ['anny', ['age', 20]]
# 从这里来看一切正常，可是anny只有18岁，重新为anny定义岁数。
anny[1][1] = 18
print(jack, tom, anny)
# ['jack', ['age', 18]] ['tom', ['age', 18]] ['anny', ['age', 18]]
# 这时候奇怪的事情发生了，jack、tom、anny的岁数都发生了改变，都变成了18了。
# jack、tom、anny应当都是不同的
print([id(x) for x in jack])
print([id(x) for x in tom])
print([id(x) for x in anny])
# [204196304, 208657608]
# [204196976, 208657608]
# [204197480, 208657608]
# 恍然大悟，原来jack、tom、anny的岁数元素指向的是同一个元素。
# 修改了其中一个，当然影响其他人了。
# 那为什么修改名称没影响呢？原来在python中字符串不可以修改，
# 所以在为tom和anny重新命名的时候，会重新创建一个’tom’和’anny’对象，
# 替换旧的’jack’对象。
```

1. **利用copy中的deepcopy方法进行深拷贝**

```python
import copy
jack = ['jack', ['age', '20']]
tom = copy.deepcopy(jack)
anny = copy.deepcopy(jack)

print(id(jack), id(tom), id(anny))# 浅拷贝的对象名称不同
# 203987272 208656392 204144456
# 从id值来看，三者是不同的对象。根据第一个思路进行重命名，重定岁数操作：
tom[0] = 'tom'
anny[0] = 'anny'
print(jack, tom, anny)
# ['jack', ['age', '20']] ['tom', ['age', '20']] ['anny', ['age', '20']]
# 从这里来看一切正常，可是anny只有18岁，重新为anny定义岁数。
anny[1][1] = 18
print(jack, tom, anny)
# ['jack', ['age', '20']] ['tom', ['age', '20']] ['anny', ['age', 18]]
# 这时候他们之间就不会互相影响了。打印出每个人的内部元素每个id：
print([id(x) for x in jack])
print([id(x) for x in tom])
print([id(x) for x in anny])
# [204194120, 203722440]
# [204197144, 203719816]
# [204196808, 203988808]
# 他们的内部元素也都指向了不同的对象。
```

注意： 对于数字，字符串和其他原子类型对象等，没有被拷贝的说法，即便是用深拷贝，查看id的话也是一样的，如果对其重新赋值，也只是新创建一个对象，替换掉旧的而已。

**例子2**

在python中，对象赋值实际上是对象的引用。当创建一个对象，然后把它赋给另一个变量的时候，python并没有拷贝这个对象，而只是拷贝了这个对象的引用。

1. **直接赋值**，传递对象的引用而已,原始列表改变，被赋值的b也会做相同的改变

```python
alist = [1,2,3,["a","b"]]
b = alist
print(b)
# [1, 2, 3, ['a', 'b']]
alist.append(5)
print(alist)
# [1, 2, 3, ['a', 'b'], 5]
print(b)
# [1, 2, 3, ['a', 'b'], 5]
print(alist is b)
# True
print(id(alist),id(b))
# 200483080 200483080
```

1. **copy浅拷贝**，只拷贝了指针对象的指针值，所以原始数据改变，子对象会改变

```python
import copy
alist = [1,2,3,["a","b"]]
c = copy.copy(alist)
print(c)
# [1, 2, 3, ['a', 'b']]
alist.append(5)
print(alist)
# [1, 2, 3, ['a', 'b'], 5]
print(c)
# [1, 2, 3, ['a', 'b']]
# alist中加了5，c中并没有加5
alist[3].append('c')
print(alist, c)
# [1, 2, 3, ['a', 'b', 'c'], 5]
# [1, 2, 3, ['a', 'b', 'c']]
# alist的第二个对象元素中加了c，
# c中的第二个对象元素中也加了c，
# 这是因为c浅复制的是第二个对象元素的指针，
# 依然会指向alist的第二个元素
print(alist is c)
# False
print(id(alist),id(c))
# 200806344 200609480
```

1. **deepcopy深拷贝**，包含对象里面的自对象的拷贝，所以原始对象的改变不会造成深拷贝里任何子元素的改变

```python
import copy
alist = [1,2,3,["a","b"]]
d = copy.deepcopy(alist)
print(d)
# [1, 2, 3, ['a', 'b']]
alist.append(5)
print(alist)
# [1, 2, 3, ['a', 'b'], 5]
print(d)
# [1, 2, 3, ['a', 'b']]
# alist中加了5，d中并没有加5
alist[3].append('c')
print(alist, d)
# [1, 2, 3, ['a', 'b', 'c'], 5]
# [1, 2, 3, ['a', 'b']]
# alist的第二个对象元素中加了c，
# c始终什么都没变，
print(alist is d)
# False
print(id(alist),id(d))
# 200806344 200609480
```

**例子3**

Python中，对象的赋值，拷贝（深/浅拷贝）之间是有差异的，如果使用的时候不注意，就可能产生意外的结果。

下面本文就通过简单的例子介绍一下这些概念之间的差别。

- **对象赋值**
- **浅拷贝**
- **深拷贝**

不同的赋值方式都用下面同一段代码，仅仅需要选择取消注释代码中如下三行的对应行注释即可

```python
#wilber = will# 对象赋值
#wilber = copy.copy(will)# 浅拷贝
#wilber = copy.deepcopy(will)# 深拷贝
```

代码如下：

```python
import copy

will = ["Will", 28, ["Python", "C#", "JavaScript"]]
#wilber = will# 对象赋值
#wilber = copy.copy(will)# 浅拷贝
#wilber = copy.deepcopy(will)# 深拷贝
print(id(will))
print(will)
print ([id(ele) for ele in will])
print (id(wilber))
print( wilber)
print ([id(ele) for ele in wilber])

will[0] = "Wilber"
will[2].append("CSS")
print (id(will))
print( will)
print ([id(ele) for ele in will])
print( id(wilber))
print (wilber)
print ([id(ele) for ele in wilber])
```

1. **对象赋值**

下面来分析一下这段代码：

- 首先，创建了一个名为will的变量，这个变量指向一个list对象，从第一张图中可以看到所有对象的地址（每次运行，结果可能不同）

- 然后，通过will变量对wilber变量进行赋值，那么wilber变量将指向will变量对应的对象（内存地址），也就是说"wilber is will"，"wilber[i] is will[i]"

  可以理解为，Python中，对象的赋值都是进行对象引用（内存地址）传递

- 第三张图中，由于will和wilber指向同一个对象，所以对will的任何修改都会体现在wilber上

  这里需要注意的一点是，str是不可变类型，所以当修改的时候会替换旧的对象，产生一个新的地址39758496

![copy_example_3_1](pic/copy_example_3_1.png)

![copy_example_3_1_1](pic/copy_example_3_1_1.jpg)

1. **浅拷贝**

分析一下这段代码：

- 首先，依然使用一个will变量，指向一个list类型的对象

- 然后，通过copy模块里面的浅拷贝函数copy()，对will指向的对象进行浅拷贝，然后浅拷贝生成的新对象赋值给wilber变量

  浅拷贝会创建一个新的对象，这个例子中**"wilber is not will"**

  但是，对于对象中的元素，浅拷贝就只会使用原始元素的引用（内存地址），**也就是说"wilber[i] is will[i]"**

- 当对will进行修改的时候

  由于list的第一个元素是不可变类型，所以will对应的list的第一个元素会使用一个新的对象39758496

  但是list的第三个元素是一个可不类型，修改操作不会产生新的对象，所以will的修改结果会相应的反应到wilber上

![copy_example_3_2](pic/copy_example_3_2.png)

![copy_example_3_2_1](pic/copy_example_3_2_1.jpg)

总结一下，当我们使用下面的操作的时候，会产生浅拷贝的效果：

- 使用切片[:]操作
- 使用工厂函数（如list/dir/set）
- 使用copy模块中的copy()函数

1. **深拷贝**

分析一下这段代码：

- 首先，同样使用一个will变量，指向一个list类型的对象

- 然后，通过copy模块里面的深拷贝函数deepcopy()，对will指向的对象进行深拷贝，然后深拷贝生成的新对象赋值给wilber变量

  跟浅拷贝类似，深拷贝也会创建一个新的对象，这个例子中**"wilber is not will"**

  但是，对于对象中的元素，深拷贝都会重新生成一份（有特殊情况，下面会说明），而不是简单的使用原始元素的引用（内存地址）

  例子中will的第三个元素指向39737304，而wilber的第三个元素是一个全新的对象39773088，也就是说，**"wilber[2] is not will[2]"**

- 当对will进行修改的时候

  由于list的第一个元素是不可变类型，所以will对应的list的第一个元素会使用一个新的对象39758496

  但是list的第三个元素是一个可不类型，修改操作不会产生新的对象，但是由于"wilber[2] is not will[2]"，所以will的修改不会影响wilber

![copy_example_3_3](pic/copy_example_3_3.png)

![copy_example_3_3_1](pic/copy_example_3_3_1.jpg)

其实，对于拷贝有一些特殊情况：

- 对于非容器类型（如数字、字符串、和其他'原子'类型的对象）没有拷贝这一说

  也就是说，对于这些类型，"obj is copy.copy(obj)" 、"obj is copy.deepcopy(obj)"

- 如果元祖变量只包含原子类型对象，则不能深拷贝，看下面的例子

![copy_example_3_4](pic/copy_example_3_4.png)



# 参考资料

- [如何理解 C++ 中的深拷贝和浅拷贝？](#如何理解 C++ 中的深拷贝和浅拷贝？)

浅拷贝深拷贝的概念和区别主要是参考的这个知乎提问中的高票答案。

- [深拷贝与浅拷贝-知乎专栏](#https://zhuanlan.zhihu.com/p/28935492)

python函数关于深浅拷贝的例子一就是参考并复制的这篇文章。

- [python的复制，深拷贝和浅拷贝的区别](http://www.cnblogs.com/xueli/p/4952063.html)

python函数关于深浅拷贝的例子二就是参考并复制的这篇文章。

- [图解Python深拷贝和浅拷贝](http://www.cnblogs.com/wilber2013/p/4645353.html)

python函数关于深浅拷贝的例子三就是参考并复制的这篇文章。



