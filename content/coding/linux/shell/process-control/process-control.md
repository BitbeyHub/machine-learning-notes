# 流程控制

- [返回顶层目录](../../../../SUMMARY.md)



# 条件语句

## if用法



### 比较命令



#### 文件比较



常用的文件比较：

```shell
-a file exists. 
-b file exists and is a block special file. 
-c file exists and is a character special file. 
-d file exists and is a directory. 
-e file exists (just the same as -a). 
-f file exists and is a regular file. 
-g file exists and has its setgid(2) bit set. 
-G file exists and has the same group ID as this process. 
-k file exists and has its sticky bit set. 
-L file exists and is a symbolic link. 
-n string length is not zero. 
-o Named option is set on. 
-O file exists and is owned by the user ID of this process. 
-p file exists and is a first in, first out (FIFO) special file or 
named pipe. 
-r file exists and is readable by the current process. 
-s file exists and has a size greater than zero. 
-S file exists and is a socket. 
-t file descriptor number fildes is open and associated with a 
terminal device. 
-u file exists and has its setuid(2) bit set. 
-w file exists and is writable by the current process. 
-x file exists and is executable by the current process. 
-z string length is zero. 
```









# 循环语句

## for用法



## while用法



## until用法



# 选择语句



## case用法



## select用法



# 参考资料

* [linux shell 流程控制（条件if,循环【for,while】,选择【case】语句实例](https://www.cnblogs.com/chengmo/archive/2010/10/14/1851434.html)

本文架构参考此博客。

* [shell脚本判断文件是否存在](https://blog.csdn.net/persever/article/details/78808356)

"文件比较"参考此博客。

