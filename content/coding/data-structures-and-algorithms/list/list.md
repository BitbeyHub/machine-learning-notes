# 链表

* [返回顶层目录](../../../SUMMARY.md)
* [返回上层目录](../data-structures-and-algorithms.md)
* [剑指offer6：从尾到头打印链表](#剑指offer6：从尾到头打印链表)
* [剑指offer22：链表中倒数第k个节点](#剑指offer22：链表中倒数第k个节点)













# 剑指offer6：从尾到头打印链表

>题目：输入一个链表的头节点，从尾到头反过来打印每个节点的值。
>
>链表节点定义如下：

```c++
struct ListNode
{
	int m_nKey;
	ListNode* m_pNext;
}
```

遍历链表是从头到尾，但是输出确实从尾到头，这就是典型的“后进先出“，可以用栈实现这个顺序。即每经过一个节点，将该节点放到一个栈中，当遍历完整个链表后，再从栈顶开始逐个输出节点的值，这时输出的节点顺序已经反过来了。

代码如下：

```c++
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        stack<int> MyStack;
        vector<int> ReturnVal;
        ListNode* p = head;
        
        while(p!=NULL) {
            MyStack.push(p->val);
            p = p->next;
        }
        
        while(!MyStack.empty()) {
            ReturnVal.push_back(MyStack.top());
            MyStack.pop();
        }
        
        return ReturnVal;
    }
};
```

# 剑指offer22：链表中倒数第k个节点











===

[面试常考-链表反转解析](https://mp.weixin.qq.com/s/zDRnSq_-pXg2iXHmVSDDbA)

