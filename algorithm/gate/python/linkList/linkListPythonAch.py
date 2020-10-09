
#单向链表结点
class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
def printLinkList(node):
    while node:
        print(node.val)
        node=node.next

if __name__=='__main__':
    l1=SingleDirectionListNode(1)
    l2=SingleDirectionListNode(2)
    l3=SingleDirectionListNode(3)
    l1.next=l2
    l2.next=l3
    printLinkList(l1)