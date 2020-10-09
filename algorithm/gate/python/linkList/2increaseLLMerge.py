'''
输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则
思路：
1. 还是先判断链表是否为空的情况
2. 从两个链表的头的val来看，谁小谁就作为函数返回的头head
3. 在借助三个指针prev，h1，h2
    prev:作为部分已经合并后的链表的尾部
    h1：link1未合并部分的头部
    h2：link2未合并部分的头部
    通过比较h1与h2的val谁小，prev的next就指向谁，
    然后prev=值小的头部
    被合并的链表头部向未被合并的部分移动
4.循环，直到h1或h2有一个为None，
5.prev.next指向未等于None的头部
6.最后返回head
'''
class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
def Merge(link1,link2):
    if link1==None and link2 ==None:
        return None
    elif link1==None:
        return link2
    elif link2==None:
        return link1
    else:
        pass
    head=link1 if link1.val<link2.val else link2#做输出用
    h1=link1
    h2=link2
    prev=head
    if h1==head:
        h1=head.next
    else:
        h2=head.next
    while h1 and h2:
        if h1.val<h2.val:
            prev.next=h1
            prev=h1
            h1=h1.next
        else:
            prev.next=h2
            prev=h2
            h2=h2.next
    if h1==None:
        prev.next=h2
    else:
        prev.next=h1
    
    return head
    
if __name__=="__main__":
    l1=SingleDirectionListNode(2)
    l2=SingleDirectionListNode(2)
    l3=SingleDirectionListNode(3)
    l4=SingleDirectionListNode(4)
    l5=SingleDirectionListNode(5)
    l6=SingleDirectionListNode(6)
    l1.next=l2
    l2.next=l3
    l4.next=l5
    l5.next=l6
    print(Merge(l1,l4).val)
    