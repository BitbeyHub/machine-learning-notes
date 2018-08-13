# 凸优化

- [返回顶层目录](../../SUMMARY.md#目录)
- [返回上层目录](numerical-calculation-and-optimization.md)

[理解凸优化](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247484439&idx=1&sn=4fa8c71ae9cb777d6e97ebd0dd8672e7&chksm=fdb69980cac110960e08c63061e0719a8dc7945606eeef460404dc2eb21b4f5bdb434fb56f92&mpshare=1&scene=1&srcid=0518tlWFr2se8CueI9c0wSTe#rd)



参考：

重要

https://blog.csdn.net/xueyingxue001/article/details/51858074



https://blog.csdn.net/shenxiaolu1984/article/details/78194053?locationNum=3&fps=1



# Jensen不等式





# 共轭函数





![conjugate-function-sup](pic/conjugate-function-sup.png)





![conjugate-function](pic/conjugate-function.png)







![conjugate-function-x4](pic/conjugate-function-x4.png)





![conjugate-function-x4-sup](pic/conjugate-function-x4-sup.png)



这里附上图的matlab代码：

```matlab
figure(1)
x = [-10:0.1:17];
y = hanshu(x);
y2 = x;
plot(x,y, 'r', x, y2, 'g', 0*x, y2, 'k', x, 0*x, 'k', x, y2 - y, 'b')
%%
axis([-7 7 -4 15])
figure(2)
axis([-7 7 -4 15])
hold on
for w1 = [-10:0.1:17]
    w0 = hanshu(w1);
    y_ge = w1*x - w0;
    plot(x, y_ge)
end
```

其中，函数y = hanshu(x)的matlab function类型代码为：

```matlab
function [y] = hanshu(x)
w0 = 2.432; w1 = 0.3675; w2 = -0.3701; w3 = 0.003058; w4 = 0.01168;
y = w0 + w1*x + w2*x.^2 + w3*x.^3 + w4*x.^4;
end
```





## 如何求共轭函数

![conjugate-function-x2](pic/conjugate-function-x2.png)



![conjugate-function-x2-sup](pic/conjugate-function-x2-sup.png)





# Fenchel不等式

