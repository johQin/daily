'''
给一个链表若其中包含环，请找到该链表环的入口结点，否则返回None
思路：
1.需要定义两个指针（快慢指针），慢指针一次向后移动一个结点，快指针一次向后移动两个结点，
2.如果链表无环，快指针会先遇到None，
3.如果链表有环，快指针会与慢指针相遇。
    假设有环的链表的结构为：0--s-->ce--m-->meet
                                 ↑<---re----↓
    fast走的距离为2x,slow走的距离为x，在环的meet结点相遇。
    ce——circle entry
    s——straight path
有：
    x=s+m
    2x=n(m+re)+s+m
得：
    s+m=n(m+re)
终：
    s=re+(n-1)(m+re)
从上面这个式子可得
如果有一个指针从链表的头部出发，一个指针从fast与slow两个点相遇的结点出发，他们会同时到达环的入口处
                                  
                                  
'''
class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
def isWithCircle(head):
    #快慢指针用来确定是否存在环
    fast=head
    slow=head
    while fast and fast.next:
        slow=slow.next
        fast=fast.next.next
        print('fast:%d,slow:%d' % (fast.val,slow.val))
        if fast==slow:
            break
    if  not (fast and fast.next):
        return None
    #确定环的入口
    tmp=head
    while tmp!=fast:
        tmp=tmp.next
        fast=fast.next
    return tmp
if __name__=="__main__":
    l1=SingleDirectionListNode(1)
    l2=SingleDirectionListNode(2)
    l3=SingleDirectionListNode(3)
    l4=SingleDirectionListNode(4)
    l5=SingleDirectionListNode(5)
    l6=SingleDirectionListNode(6)
    l7=SingleDirectionListNode(7)
    l1.next=l2
    l2.next=l3
    l3.next=l4
    l4.next=l5
    l5.next=l6
    l6.next=l7
    l7.next=l4
    print('是否有环，环的位置在%s' % (isWithCircle(l1).val if isWithCircle(l1) else None))

        
