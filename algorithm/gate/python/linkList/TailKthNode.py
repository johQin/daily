'''
输入一个链表，输出该链表中倒数第k个节点
思路：
1.借助两个指针left和right，初始化两个指针为头结点
2.然后让两个指针，间隔k个距离，如果链表比k短，那么直接返回None
3.然后让间隔k个距离的两个指针，同时向后移动，直到right指针为None，返回left

'''
class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
def TailKthNode(startNode,k):
    if not isinstance(startNode,SingleDirectionListNode):
        return None
    left=startNode
    right=startNode
    for i in range(k):
        if not right:
            return None
        right=right.next
    while right!=None:
        right=right.next
        left=left.next
    return left.val

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
    print(TailKthNode(l1,3))