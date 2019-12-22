# 递归

* [返回顶层目录](../../../../SUMMARY.md)
* [返回上层目录](../jianzhi-offer.md)



# 剑指offer6：从尾到头打印链表

>题目：输入一个链表的头节点，从尾到头反过来打印每个节点的值。
>
>链表节点定义如下：
>
>```c++
>struct ListNode
>{
>	int m_nKey;
>	ListNode* m_pNext;
>}
>```

遍历链表是从头到尾，但是输出确实从尾到头，这就是典型的“后进先出“，可以用栈实现这个顺序。即每经过一个节点，将该节点放到一个栈中，当遍历完整个链表后，再从栈顶开始逐个输出节点的值，这时输出的节点顺序已经反过来了。

c++:

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

[详情](https://cuijiahua.com/blog/2017/11/basis_3.html)，[练习](https://www.nowcoder.com/practice/d0267f7f55b3412ba93bd35cfa8e8035?tpId=13&tqId=11156&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)。



# 参考资料

- [剑指Offer系列刷题笔记汇总](https://cuijiahua.com/blog/2018/02/basis_67.html)

本文参考此博客。

