'''
输入一个复杂链表，链表的每一个结点包含val，next（指向下一个结点），random（指向一个任意的结点）
克隆这个复杂链表，返回原链表的副本。
思路：
1.复制每一个结点形成副本，并且将副本紧插入到原结点的后面，暂时不做random的指向工作
2.实现新建node的random指向
3.断开副本与原本的链接
'''

class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
        self.random=None
def Clone(startNode):
    
    #复制每一个结点形成副本，并且将副本紧插入到原结点的后面，暂时不做random的指向工作
    tmp=startNode
    while tmp:
        node=SingleDirectionListNode(tmp.val)
        node.next=tmp.next
        tmp.next=node
        tmp=node.next
    
    #实现新建node的random指向
    tmp=startNode
    while tmp:
        if tmp.random:
            tmp.next.random=tmp.random.next
        #tmp.next作为tmp的副本，tmp.next.ramdom是副本的random
        #tmp作为原本，tmp.random作为原本的random，tmp.random.next作为原本的random的副本
        #将原本的random的副本赋值给原本的副本的random
        #这解决的副本random的指向问题
        tmp=tmp.next.next
    
    #断开副本与原本的链接
    tmp=startNode
    tmpr=startNode.next
    res=startNode.next
    while tmpr:
        tmp.next=tmpr.next
        tmp=tmpr
        tmpr=tmpr.next
    
    return res

    
if __name__=="__main__":
    l1=SingleDirectionListNode(1)
    l2=SingleDirectionListNode(2)
    l3=SingleDirectionListNode(3)
    l4=SingleDirectionListNode(4)
    l5=SingleDirectionListNode(5)
    l6=SingleDirectionListNode(6)
    l1.next=l2
    l1.random=l3
    l2.next=l3
    l2.random=l4
    l3.next=l4
    l3.random=l5
    l4.next=l5
    l4.random=l6
    l5.next=l6
    l5.random=l3
    l6.random=None
    res=Clone(l1)
    while res:
        print('结点值：{0},结点的随机值:{1}'.format(res.val,res.random.val if res.random else None))
        res=res.next