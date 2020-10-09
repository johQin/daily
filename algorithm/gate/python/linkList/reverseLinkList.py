'''
输入一个链表，反转链表后，输出新链表的表头
思路：
1.当链表结点只有一个时，直接返回该结点
2.当链表有多个结点时，采用三个指针left，node，right
3.将三个指针分别初始化，left=startNode,node=start.next,right=node.next,并将首结点的next赋值为None
4.然后将node.next指向left，node与right之间的链接就断掉了，此后依次将left、node、right往后移
5.直到right为空，跳出循环，跳出之后，需要将node.next=left
'''
class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
def reverse(startNode):
    node=startNode
    if node==None:
        return None
    if node.next==None:
        return node
    left=node
    node=node.next
    right=node.next
    left.next=None
    while right!=None:
        node.next=left
        left=node
        node=right
        right=right.next
    node.next=left
    return node



if __name__=="__main__":
    l1=SingleDirectionListNode(1)
    l2=SingleDirectionListNode(2)
    l3=SingleDirectionListNode(3)
    l4=SingleDirectionListNode(4)
    l5=SingleDirectionListNode(5)
    l6=SingleDirectionListNode(6)
    l1.next=l2
    l2.next=l3
    l3.next=l4
    l4.next=l5
    l5.next=l6
    print(reverse(l1).val)