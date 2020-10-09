
'''
输入两个链表，找出它们第一个公共结点。
第一个公共结点之后，两个链表的结点就相同了
思路：
1.首先借助两个指针分别指向两个链表
2.找出它们的链表长度差k，然后让长的那个链表的指针先走到第k个位置。
3.由于第一个公共结点之后，两个链表的结点就相同了。
    从链表末尾的角度来看，较长的链表的指针处在结点k的位置上，较短的那个链表的指针处在开头。
    如果同时后移这两个指针，他们两个会同时在第一个公共结点处相遇，
    如果到链表的末尾处，这两个指针都没有指向同一个结点，那么便不存在公共的结点
'''
class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
def findLongerk(link1,link2):
    l1=link1
    l2=link2
    while l1 and l2:#一旦有一个链表的指针走到末尾，那么就跳出
        l1=l1.next
        l2=l2.next
    k=0
    #判断哪一个链表较长
    if l1:#链表较长的那一个继续向后移动，并计数，直到移到链表末尾，那么k就为两个链表的长度之差
        while l1:
           k+=1
        return find1thComNode(link1,link2,k)
    else:
        while l2:
            k+=1
        return find1thComNode(link2,link1,k)
def find1thComNode(long,short,k):
    for i in range(k):#让长的那个指针先向后移k步
        long=long.next
    while long!=short:#一旦两个指针同时指向同一个结点（第一个公共结点），就跳出循环。
        long=long.next
        short=short.next
    return long#返回指针所指向的结点



if __name__=='__main__':
    l1=SingleDirectionListNode(1)
    l2=SingleDirectionListNode(2)
    l3=SingleDirectionListNode(3)
    l4=SingleDirectionListNode(4)
    l5=SingleDirectionListNode(5)
    l6=SingleDirectionListNode(6)
    l7=SingleDirectionListNode(7)
    l1.next=l2
    l2.next=l5
    l3.next=l4
    l4.next=l5
    l5.next=l6
    l6.next=l7
    com=findLongerk(l1,l3)
    print('公共结点值%s' % com.val if com else None)