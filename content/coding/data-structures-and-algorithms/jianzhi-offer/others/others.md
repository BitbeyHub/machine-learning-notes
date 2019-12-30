# 其他

* [返回顶层目录](../../../../SUMMARY.md)
* [返回上层目录](../jianzhi-offer.md)
* [剑指offer14：剪绳子](#剑指offer14：剪绳子)



# 剑指offer15：二进制中1的个数

> 题目：输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

不右移输入的数字n，左移数字1.

c++:

```c++
class Solution {
public:
     int  NumberOf1(int n) {
         int count = 0;
         unsigned int flag = 1;
         while(flag){
             if(n & flag){
                 count++;
             }
             
             flag = flag << 1;
         }
         
         return count;
     }
};
```

[详情](https://cuijiahua.com/blog/2017/11/basis_11.html)，[练习](https://www.nowcoder.com/practice/8ee967e43c2c4ec193b040ea7fbb10b8?tpId=13&tqId=11164&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)。



# 剑指offer16：数值的整数次方

> 题目：输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

不右移输入的数字n，左移数字1。

c++:

```c++
class Solution {
    bool g_InvalidInput = false;
    
public:
    double Power(double base, int exponent) {
        g_InvalidInput = false;
        if(equal(base, 0.0) && exponent < 0) {
            g_InvalidInput = true;
            return 0.0;
        }
        unsigned int absExponent = (unsigned int)(exponent);
        if(exponent < 0) {
            absExponent = (unsigned int)(-exponent);
        }
        
        double result = PowerWithUnsignedExponent(base, absExponent);
        if(exponent < 0) {
            result =  1.0 / result;
        }
        return result;
    }
private:
    double PowerWithUnsignedExponent(double base, unsigned int exponent) {
        double result = 1.0;
        for(int i = 1; i <= exponent; i++) {
            result *= base;
        }
        return result;
    }
    
    bool equal(double num1, double num2){
        if(num1 - num2 > -0.0000001 && (num1 - num2) < 0.0000001){
            return true;
        } else{
            return false;
        }
    }
};
```

[详情](https://cuijiahua.com/blog/2017/11/basis_12.html)，[练习](https://www.nowcoder.com/practice/1a834e5e3e1a4b7ba251417554e07c00?tpId=13&tqId=11165&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)。



# 剑指offer29：顺时针打印矩阵

> 题目：输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下矩阵：
>
> ![others-29](pic/others-29.jpg)
>
> 则依次打印出数组：1，2，3，4，8，12，16，15，14，13，9，5，6，7，11，10。

将结果存入vector数组，从左到右，再从上到下，再从右到左，最后从下到上遍历。

c++:

```c++
class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {
        int rows = matrix.size();			//行数
        int cols = matrix[0].size();		//列数
        vector<int> result;
        
        if(rows == 0 && cols == 0){
            return result;
        }
        int left = 0, right = cols - 1, top = 0, bottom = rows - 1;
        
        while(left <= right && top <= bottom){
            //从左到右
            for(int i = left; i <= right; ++i){
                result.push_back(matrix[top][i]);
            }
            //从上到下
            for(int i = top + 1; i <= bottom; ++i){
                result.push_back(matrix[i][right]);
            }
            //从右到左
            if(top != bottom){
                for(int i = right - 1; i >= left; --i){
                    result.push_back(matrix[bottom][i]);
                }
            }
            //从下到上
            if(left != right){
                for(int i = bottom - 1; i > top; --i){
                    result.push_back(matrix[i][left]);
                }
            }
            left++, top++, right--, bottom--;
        }
        return result;
    }
};
```

[详情](https://cuijiahua.com/blog/2017/12/basis_19.html)，[练习](https://www.nowcoder.com/practice/9b4c81a02cd34f76be2659fa0d54342a?tpId=13&tqId=11172&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)。







# 参考资料

- [面试题：剪绳子──动态规划 or 贪心算法](https://blog.csdn.net/sinat_36161667/article/details/80785142)

本文参考此博客。

