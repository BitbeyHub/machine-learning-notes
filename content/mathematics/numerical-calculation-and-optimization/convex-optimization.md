# 凸优化

- [返回顶层目录](../../SUMMARY.md#目录)
- [返回上层目录](numerical-calculation-and-optimization.md)

[理解凸优化](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247484439&idx=1&sn=4fa8c71ae9cb777d6e97ebd0dd8672e7&chksm=fdb69980cac110960e08c63061e0719a8dc7945606eeef460404dc2eb21b4f5bdb434fb56f92&mpshare=1&scene=1&srcid=0518tlWFr2se8CueI9c0wSTe#rd)



参考：

重要

https://blog.csdn.net/xueyingxue001/article/details/51858074



https://blog.csdn.net/shenxiaolu1984/article/details/78194053?locationNum=3&fps=1



# 凸集



## Jensen不等式



# 凸函数



## 共轭函数





![conjugate-function-sup](pic/conjugate-function-sup.png)





![conjugate-function](pic/conjugate-function.png)

注意，这个函数并不是凸函数。





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



### 如何求共轭函数

![conjugate-function-x2](pic/conjugate-function-x2.png)



![conjugate-function-x2-sup](pic/conjugate-function-x2-sup.png)





## Fenchel不等式







# 凸优化问题







# 对偶



## 拉格朗日函数

如上一章节所示，标准形式的优化问题如下：
$$
\begin{aligned}
&\text{minimize}\quad &f_0(x)\quad&\\
&\text{subject to}\quad &f_i(x)\leqslant0,\quad &i=1,...,m\\
&&h_i(x)=0,\quad&i=1,...,p
\end{aligned}
$$
其中，自变量x∈R^n，设问题的定义域为
$$
\mathbb{D}=\bigcap_{i=0}^{m}\text{dom}\ f_i\ \cap\ \bigcap_{i=1}^{p}\text{dom}\ h_i
$$
是非空集合，优化问题的最优值为p\*。注意到这里并没有假设上述问题是凸优化问题。

Lagrange对偶的基本思想是在目标函数中考虑上述问题的约束条件，即添加约束条件的加权和，得到增广的目标函数。

定义上述问题的**Lagrange函数**L为
$$
L(x,\lambda,\nu)=f_0(x)+\sum_{i=1}^m\lambda_if_i(x)+\sum_{i=1}^p\nu_ih_i(x)
$$
其中，λi称为第i个不等式约束fi(x)≤0对应的**Lagrange乘子**；类似地，νi称为第i个等式约束hi(x)=0对应的Lagrange乘子。向量λ和ν称为对偶函数的变量，或者称为上述问题的Lagrange乘子向量。

## 拉格朗日对偶函数

定义**Lagrange对偶函数**g为Lagrange函数关于x取得的最小值，即对任意的λ和ν，有
$$
\begin{aligned}
g(\lambda,\nu)&=\mathop{\text{inf}}_{x\in \mathbb{D}}L(x,\lambda,\nu)\\
&=\mathop{\text{inf}}_{x\in \mathbb{D}}\left( f_0(x)+\sum_{i=1}^m\lambda_if_i(x)+\sum_{i=1}^p\nu_ih_i(x) \right)
\end{aligned}
$$
对上式的不严谨解释：保持λ和ν不变，找到拉格朗日函数L的下确界（最小值），即为g(λ,ν)函数的值。

### 最优值的下界

对偶函数构成了原问题最优值p\*的下界：即对任意λ≥0和ν，下式都成立：
$$
g(\lambda,\nu)\leqslant p^*
$$
下面来很容易地验证这个重要性质。

假设
$$
\tilde{x}
$$
是原问题的一个可行点，即
$$
f_i(\tilde{x})\leqslant 0,\quad h_i(\tilde{x})=0
$$
。根据假设，λ≥0，我们有
$$
\sum_{i=1}^m\lambda_if_i(\tilde{x})+\sum_{i=1}^p\nu_ih_i(\tilde{x})\leqslant0
$$
这是因为左边的第一项非正，而第二项为零。根据上述不等式，有
$$
L(\tilde{x},\lambda,\nu)=f_0(\tilde{x})+\sum_{i=1}^m\lambda_if_i(\tilde{x})+\sum_{i=1}^p\nu_ih_i(\tilde{x})\leqslant f_0(\tilde{x})
$$
由于每一个可行点
$$
\tilde{x}
$$
都满足
$$
g(\lambda,\nu)\leqslant f_0(\tilde{x})
$$
，因此不等式
$$
g(\lambda,\nu)\leqslant p^*
$$
成立。针对x∈R和具有一个不等式约束的某简单问题，下图描述了上式所给出的下界。

![dual-lower-bound](pic/dual-lower-bound.png)

上图为对偶可行点给出的下界。实线表示目标函数f0，虚线表示约束函数f1。可行集是区间[-0.46,0.46]，如图中两条垂直点线所示。最优点和最优值分别为x\*=-0.46，p\*=1.54（在图中用红色原点表示）。点线表示一些列Lagrange函数L(x,λ)，其中λ=0.1,0.2,...,1.0。每个Lagrange函数都有一个极小值，均小于原问题最优目标值p\*，这是因为在可行集上（假设λ≥0）有L(x,λ)≤f0(x)。

### 通过线性逼近来理解下界性质

可通过示性函数进行线性逼近来理解拉格朗日函数和其给出下界的性质。

首先将原问题重新描述为一个无约束问题
$$
\text{minimize}\quad f_0(x)+\sum_{i=1}^mI\text{_}(f_i(x))+\sum_{i=1}^pI_0(h_i(x))
$$
其中，I_是非正实数集的示性函数
$$
\begin{aligned}
I\text{_}(u)=
\left\{\begin{matrix}
&0\quad &u\leqslant 0\\ 
&\infty\quad &u>0
\end{matrix}\right.
\end{aligned}
$$
类似地，I0是集合\{0\}的示性函数。在上面的无约束表达式中，函数I\_(u)可以理解为我们对约束函数值u=fi(x)的一种恼怒或不满：如果fi(x)≤0，I\_(u)为零，如果fi(x)>0，I\_(u)为∞。用砖块撞墙来理解比较直观：当砖块还没撞墙，接触力为0，当砖块撞进了墙里面，接触力为无穷大。类似地，I0(u)表达了我们对等式约束值u=hi(x)的不满。我们可以认为函数I\_是一个“砖墙式”或“无限强硬”的不满意方程；即随着函数fi(x)从非正数变为正数，我们的不满意从零升到无穷大。

假设在上面的表达式中，用线性函数λiu替代函数I\_(u)，其中λi≥0，用函数νiu替代I0(u)。则目标函数变为拉格朗日函数L(x,λ,ν)，且对偶函数值g(λ,ν)是问题
$$
L(x,\lambda,\nu)=f_0(x)+\sum_{i=1}^m\lambda_if_i(x)+\sum_{i=1}^p\nu_ih_i(x)
$$
的最优值。在上述表达式中，我们用线性或者“软”的不满意函数替换了函数I\_和I0。对于不等式约束，如果fi(x)=0，我们的不满意度为零，当fi(x)>0，我们的不满意度大于零；随着约束“越来越被违背”，我们越来越不满意。在上面的原始表达式中，任意不大于零的fi(x)都是可接受的，而在软的表达式中，当约束有裕量时，我们会感到满意，例如当fi(x)<0时。

显然，用线性函数λiu去逼近I\_(u)是远远不够的。然而，线性函数至少可以看成是示性含糊的一个下估计。这是因为对任意u，有
$$
\lambda_iu\leqslant I\text{_}(u),\quad \nu_iu\leqslant I_0(u)
$$
，我们随之可以得到，**对偶函数是原问题最优函数值的一个下界**。

### 拉格朗日对偶函数和共轭函数

前面“共轭函数”一节提到的函数f的共轭函数f*为
$$
f^*(y)=\mathop{\text{sup}}_{x\in dom\ f}(y^Tx-f(x))
$$
事实上，共轭函数和拉格朗日对偶函数紧密相关。下面的问题说明了一个简单的联系，考虑问题
$$
\begin{aligned}
&\text{minimize}\quad &f(x)\\
&\text{subject to}\quad &x=0\\
\end{aligned}
$$
（虽然此问题没有什么挑战性，目测就可以看出答案）。上述问题的拉格朗日函数为
$$
L(x,\nu)=f(x)+\nu^Tx
$$
，其对偶函数为
$$
\begin{aligned}
g(\nu)=&\mathop{\text{inf}}_x\left(f(x)+\nu^Tx\right)\\
=&-\mathop{\text{sup}}_x\left((-\nu)^Tx-f(x)\right)\\
=&-f^*(-\nu)
\end{aligned}
$$
更一般地（也更有用地），考虑一个优化问题，其具有线性不等式以及等式约束，
$$
\begin{aligned}
&\text{minimize}\quad &f_0(x)\\
&\text{subject to}\quad &Ax\leqslant b\\
&&Cx=d\\
\end{aligned}
$$
利用函数f0的共轭函数，我们可以将问题（上式）的对偶函数表述为
$$
\begin{aligned}
g(\lambda,\nu)=&\mathop{\text{inf}}_x\left( f_0(x)+\lambda^T(Ax-b)+\nu^T(Cx-d) \right)\\
=&-b^T\lambda-d^T\nu+\mathop{\text{inf}}_x\left( f_0(x)+(A^T\lambda+C^T\nu)^Tx \right)\\
=&-b^T\lambda-d^T\nu-f^*_0(-A^T\lambda-C^T\nu)
\end{aligned}
$$
函数g的定义域也可以由函数f0*的定义域得到：
$$
\text{dom}\ g=\left\{ (\lambda,\nu)|-A^T\lambda-C^T\nu\in\text{dom}\ f_0^* \right\}
$$

## 拉格朗日对偶问题

对于任意一组(λ,ν)，其中λ≥0，Lagrange对偶函数给出了优化问题的最优值的一个下界。因此，我们可以得到和参数λ、ν相关的一个下界。一个自然的问题是：从Lagrange函数能够得到的**最好**下界是什么？

可以讲这个问题表述为优化问题
$$
\begin{aligned}
&\text{maximize}\quad &g(\lambda,\nu)\\
&\text{subject to}\quad &\lambda\geqslant0
\end{aligned}
$$
上述问题称为问原始问题的**Lagrange对偶问题**。原始问题也被称为原问题。若解(λ\*,ν\*)是对偶问题的最优解，则称解(λ\*,ν\*)是**对偶最优解**或者是**最优Lagrange乘子**。

Lagrange对偶问题是一个凸优化问题，这是因为极大化的目标函数是凹函数（因为它包含负的共轭函数，而共轭函数是凸函数），且约束集合是凸集。因此，对偶问题的凸性和原问题是否是凸优化问题无关。

### 弱对偶性

Lagrange对偶问题的最优值，我们用d\*表示，根据定义，这是通过Lagrange函数得到的原问题最优值p\*的最好下界。特别地，我们有下面简单但是非常不重要的不等式
$$
d^*\leqslant p^*
$$
即使原问题不是凸问题，上述不等式依然成立。这个性质称为**弱对偶性**。

定义差值p\*-d\*是原问题的最优对偶间隙。它给出了原问题最优值以及通过Lagrange对偶函数所能得到的最好（最大）下界之间的差值。罪有对偶间隙总是非负的。

**当原问题很难求解时，弱对偶不等式（上式）可以给出原问题最优值的一个下界，这是因为对偶问题总是凸问题，而且在很多情况下都可以进行有效的求解得到d\***。

### 强对偶性和Slater约束准则

如果等式
$$
d^*=p^*
$$
成立，即最优对偶间隙为零，那么强对偶性成立。这说明从Lagrange对偶函数得到的最好下界是紧的。

对于一般情况，强对偶性不成立。但是，如果原问题是凸问题，即可以表述为如下形式
$$
\begin{aligned}
&\text{minimize}\quad &f_0(x)\quad&\\
&\text{subject to}\quad &f_i(x)\leqslant0,\quad &i=1,...,m\\
&&Ax=b,\quad&\\
\end{aligned}
$$
其中，函数f0,...,fm是凸函数，强对偶性通常（但不总是）成立。有很多研究成果给除了强对偶性成立的条件（除了凸性条件外），这些条件称为**约束准则**。

一个简单的约束准则是**Slater条件**：存在一点x∈relint D（relint：relative interior相对内部，即D的相对内点集）使得下式成立
$$
f_i(x)<0,\quad i=1,...,m,\quad Ax=b
$$
满足上述条件的点有时称为**严格可行**，这是因为不等式约束严格成立。

Slater条件是说：存在x，使不等式约束中的“小于等于号”要严格取到“小于号”。

Slater定理说明，**当Slater条件成立（且原问题是凸问题）时，强对偶性成立。**

当不等式约束函数fi中有一些是仿射函数时，Slater条件可以进一步改进。如果最前面的k个约束函数f1,...,fk是仿射的，则若下列弱化的条件成立，强对偶性成立。该条件为：存在一点x∈relint D，使得
$$
\begin{aligned}
&f_i(x)\leqslant 0,\quad i=1,...,k\\
&f_i(x)<0,\quad i=k+1,...,m\\
&Ax=b
\end{aligned}
$$
换言之，仿射不等式不需要严格成立。注意到当所有约束条件都是线性等式或不等式且dom f0是开集时，改进的Slater条件就是可行性条件。

若Slater条件（以及其改进形式）满足，则对于凸问题，强对偶性成立，即存在一组对偶可行解(λ\*,ν\*)使得g(λ\*,ν\*)=d\*=p\*。

## 拉格朗日对偶的解释

### 强弱对偶性的极大极小描述

可以将原、对偶优化问题以一种**更为对称**的方式进行表达，为了简化讨论，假设没有等式约束；事实上，现有的结果很容易就能拓展到有等式约束的情形。

首先，我们注意到
$$
\begin{aligned}
\mathop{\text{sup}}_{\lambda\geqslant 0}L(x,\lambda)&=\mathop{\text{sup}}_{\lambda\geqslant 0}\left( f_0(x)+\sum_{i=1}^m\lambda_if_i(x) \right)\\
&=\left\{\begin{matrix}
f_0(x)&f_i(x)\leqslant0,\quad i=1,...,m\\ 
\infty&\text{other}
\end{matrix}\right.
\end{aligned}
$$
假设x不可行，即存在某些i使得fi(x)>0。选择λj=0，j≠i，以及λi→∞，可以得出
$$
\mathop{\text{sup}}_{\lambda\geqslant 0}L(x,\lambda)=\infty
$$
。反过来，如果x可行，则有fi(x)≤0,i=1,...,m，λ的最优选择为λ=0，
$$
\mathop{\text{sup}}_{\lambda\geqslant 0}L(x,\lambda)=f_0(x)
$$
。这意味着我们可以**将原问题的最优值写成如下形式**：





















### 鞍点解释





### 对策解释





## 最优性条件

###互补松弛性

设原问题和对偶问题的最优值都可以达到且相等（即强对偶性成立）。令x\*是原问题的最优解，(λ\*,ν\*)是对偶问题的最优解，这表明
$$
\begin{aligned}
f_0(x^*)&=g(\lambda^*,\mu^*)\\
&=\mathop{\text{inf}}_x\left( f_0(x)+\sum_{i=1}^m\lambda_i^*f_i(x)+\sum_{i=1}^p\mu_i^*h_i(x) \right)\\
&\leqslant f_0(x^*)+\sum_{i=1}^m\lambda_i^*f_i(x^*)+\sum_{i=1}^p\mu_i^*h_i(x) \\
&\leqslant f_0(x^*)
\end{aligned}
$$
第一个等式说明最优对偶间隙为零，第二个等式是对偶函数的定义。第三个不等式是根据Lagrange函数关于x求下确界小于等于其在x=x\*处的值得来。最后一个不等式的成立是因为下式
$$
\begin{aligned}
&\lambda_i^*\geqslant0,\quad f_i(x^*)\leqslant0,\ &i=1,...,m\\
&h_i(x^*)=0,&i=1,...,p
\end{aligned}
$$
。因此，在上面的式子链中，两个不等式取等号。

可以由此得出一些有意义的结论。例如，由于第三个不等式变为等式，我们知道L(x,λ\*,ν\*)关于x求极小时在x\*处取得最小值（Lagrange函数L(x,λ\*,ν\*)也可以有其他最小点；x\*只是其中一个最小点）。

另外一个重要的结论是
$$
\sum_{i=1}^m\lambda_i^*f_i(x^*)=0
$$
事实上，求和项的每一项都非正，因此有
$$
\lambda_i^*f_i(x^*)=0,\quad i=1,...,m
$$
上述条件成为**互补松弛性**；它对任意原问题最优解x\*以及对偶问题最优解(λ\*,ν\*)都成立（当强对偶性成立时）。我们可以将互补松弛条件写成
$$
\lambda_i^*>0\rightarrow f_i(x^*)=0
$$
或者等价地
$$
f_i(x^*)<0\rightarrow \lambda_i^*=0
$$
粗略地讲，上式意味着在最优点处，除了第i个约束起作用的情况（fi(x\*)=0），最优Lagrange乘子的第i项都为零。

### KKT最优性条件

现在假设函数f0,...,fm,h1,...,hp可微（因此定义域是开集），但并不假设这些函数是凸函数。

**非凸问题的KKT条件**

和前面一样，令x\*和(λ\*,ν\*)分别是原问题和对偶问题的某对最优解，**对偶间隙为零（需要满足Slater条件）**。因为L(x,λ\*,ν\*)关于x求极小在x\*处取得最小值，因此函数在x*处的导数必须为零，即
$$
\triangledown f_0(x^*)+\sum_{i=1}^m\lambda_i^*\triangledown f_i(x^*)+\sum_{i=1}^p\mu_i^*\triangledown h_i(x^*)=0
$$
因此，我们有
$$
\begin{aligned}
f_i(x^*)\leqslant0&,\quad i=1,...,m\\
h_i(x^*)\leqslant0&,\quad i=1,...,p\\
\lambda_i^*\geqslant0&,\quad i=1,...,m\\
\lambda_i^*f_i(x^*)=0&,\quad i=1,...,m\\
\triangledown f_0(x^*)+\sum_{i=1}^m\lambda_i^*\triangledown f_i(x^*)+\sum_{i=1}^p\mu_i^*\triangledown h_i(x^*)=0&\\
\end{aligned}
$$
我们称上式为**Karush-Kuhn-Tucker**（KKT）条件。

总之，对于目标函数和约束函数可微的任意优化问题，如果**强对偶性成立**，那么任意一对原问题最优解和对偶问题最优解必须满足KKT条件。即KKT条件是一组解成为最优解的**必要条件**。

**凸问题的KKT条件**

**当原问题是凸问题时，满足KKT条件的点也是原、对偶最优解**，即KKT条件是一组解成为最优解的充分条件。换言之，如果函数fi是凸函数，hi是放射函数，x\*、λ\*、ν\*是以满足KKT条件的点，那么，x\*和(λ\*,ν\*)分别是原问题和对偶问题的最优解，对偶间隙为零。

总结上面两段：当原问题是凸问题时，KKT条件是一组解成为最优解的**充分条件**；当强对偶性成立，KKT条件是一组解成为最优解的**必要条件**。

**所以，若某个凸优化问题满足Slater条件，那么KKT条件是最优性的充要条件：Slater条件意味着最优对偶间隙为零，且对偶最优解可以达到，因此x是原问题的最优解，当且仅当存在(λ,ν)，二者满足KKT条件。**

也就是说，当一个问题满足下面两个条件：

* 原始问题是凸的，即凸优化问题
* 满足Slater条件

那么，(x\*,λ\*,ν\*)满足KKT条件，等价于（充要条件）x\*是原问题的最优解。

KKT条件的用途 ：

KKT条件在优化领域有着重要作用。在一些特殊的情况下，是可以求解KKT条件的（因此也可以求解优化问题）。更一般地，很多求解凸优化问题的方法可以认为或者理解为求解KKT条件的方法。

KKT条件可以用于如下方面：

* 有时候可以直接从KKT条件里得到最优的解析解。 
* 等式约束的优化问题，可以通过KKT条件转化为无约束方程求零点问题。 
* 有不等式约束的优化问题，可以使用KKT条件来简化，帮助求解。

### KKT条件的力学解释

可以从力学角度（这其实也是最初提出Lagrange的动机）对KKT条件给出的一个较好的解释。我们可以通过一个简单的例子描述这个想法。下图所示系统包含两个连在一起的模块，左右两端是墙，通过三段弹簧将它们连在一起。

模块的位置用x描述，x1是左边模块的中心点的位移，x2是右边模块中心点的位移。左边墙的位置是0，右边墙的位置是l。

模块本身的宽度是w>0，且它们之间不能互相穿透，也不能穿透墙。

![KKT-spring-explanation](pic/KKT-spring-explanation.png)

弹性势能可以写成模块位置的函数
$$
f_0(x_1,x_2)=\frac{1}{2}k_1x_1^2+\frac{1}{2}k_2(x_2-x_1)^2+\frac{1}{2}(l-x_2)^2
$$
其中，ki>0是三段弹簧的劲度系数。在满足以下不等式的约束
$$
\begin{aligned}
w/2-x_1&\leqslant0\\
w+x_1-x_2&\leqslant0\\
w/2-l+x_2&\leqslant0
\end{aligned}
$$
的条件下极小化弹性使能可以得到平衡位置x*。这些约束也称为运动约束，它描述了模块的宽度w>0，且不同的模块之间以及模块和墙之间不能穿透，通过求解如下优化问题可以得到平衡位置
$$
\begin{aligned}
&\text{minimize}\quad &\left(1/2)(k_1x_1^2+k_2(x_2-x_1)^2+k_3(l-x_2)^2\right)\\
&\text{subjuect to}\quad &w/2-x_1\leqslant0\\
&&w+x_1-x_2\leqslant0\\
&&w/2-l+x_2\leqslant0
\end{aligned}
$$
这是一个二次规划问题。

引入Lagrange乘子λ1，λ2，λ3，此问题的KKT条件包含：

* 运动约束
  $$
  \begin{aligned}
  w/2-x_1&\leqslant0\\
  w+x_1-x_2&\leqslant0\\
  w/2-l+x_2&\leqslant0
  \end{aligned}
  $$

* 非负约束
  $$
  \lambda_i\geqslant0
  $$

* 互补松弛条件
  $$
  \begin{aligned}
  \lambda_1(w/2-x_1)&=0\\
  \lambda_2(w+x_1-x_2)&=0\\
  \lambda_3(w/2-l+x_2)&=0
  \end{aligned}
  $$

* 零梯度条件
  $$
  \begin{aligned}
  \begin{bmatrix}
   k_1x_1-k_2(x_2-x_1)\\ 
   k_2(x_2-x_1)-k_3(l-x_2)
  \end{bmatrix}
  +\lambda_1
  \begin{bmatrix}
  -1\\ 
  0
  \end{bmatrix}
  +\lambda_2
  \begin{bmatrix}
  1\\ 
  -1
  \end{bmatrix}
  +\lambda_3
  \begin{bmatrix}
  0\\ 
  1
  \end{bmatrix}
  =0
  \end{aligned}
  $$



上式可以理解为两个模块间的受力平衡方程，这里假设**Lagrange乘子是模块之间，模块与墙之间的接触力**，如下图所示。第一个方程表示第一个模块上的总受力为零，其中λ1是左边墙事假在这个模块上的接触力，-λ2是右边模块给的力。**当存在接触时，接触力不为零（如上面的互补松弛条件所描述）**，上面的互补松弛条件中的最后一个条件表明，除非右边模块接触墙，否则λ3为零。

![spring-force-analyze](pic/spring-force-analyze.png)

在这个例子中，弹性势能和运动约束方程都是凸函数，若2w≤l且Slater约束准则成立，即墙之间有足够的孔家安放两个模块，我们有：原始问题（即求弹性势能最小）的平衡点能量表述和KKT条件给出的受力平衡表述具有一样的结果。

### 通过解对偶问题求解原问题

前面已经提到，如果**强对偶性**成立，且存在一个对偶最优解(λ\*,ν\*)，那么任意原问题最优点也是L(X,λ\*,ν\*)的最优解（为什么？简单理解：当λ≠0时，f=0，当λ=0时，f≤0，而h=0一直满足）。这个性质可以让我们从对偶最优方程中去求解原问题最优解。

更精确地，假设强对偶性成立，对偶最优解(λ\*,ν\*)已知。假设L(X,λ\*,ν\*)的最小点，即下列问题的解
$$
\text{minimize}\quad f_0(x)+\sum_{i=1}^m\lambda_i^*f_i(x)+\sum_{i=1}^p\nu_i^*h_i(x)
$$
唯一。那么，如果上式的解是原问题的可行解，那么它一定就是原问题的最优解；如果它不是原问题的可行解，那原问题本身就不存在最优解，即原问题的最优解无法达到。

**当对偶问题比原问题更容易求解时**，比如说对偶问题可以解析求解或者有某些特殊的结构容易求解，上述方法很有效。

