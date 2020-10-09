'''
输入一个链表，按链表值从尾到头的顺序返回一个数组
'''
#单向链表结点
class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
    def toString(self):
        return '(val:{0},next:{1})'.format(self.val,self.next)
def LinkListT2H(startNode):
    if not isinstance(startNode,SingleDirectionListNode):
        return None
    target=[]
    node=startNode
    while node:
        target.append(node.val)
        node=node.next
    target.reverse()
    return target 
if __name__=="__main__":
    l1=SingleDirectionListNode(1)
    l2=SingleDirectionListNode(2)
    l3=SingleDirectionListNode(3)
    l1.next=l2
    l2.next=l3
    print('从尾到头的顺序返回一个数组',LinkListT2H(l1))
