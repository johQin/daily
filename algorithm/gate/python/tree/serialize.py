'''
实现序列化和反序列二叉树
'''
def serialize(root):
    tmpNode=root
    stack=[]
    res=[]
    while tmpNode or stack:
        while tmpNode:
            res.append(str(tmpNode.val))
            stack.append(tmpNode)
            tmpNode=tmpNode.left
        res.append('#')
        tmpNode=stack.pop()
        tmpNode=tmpNode.right
    return ' '.join(res)
def deserialize(strs):
    strlist=strs.split(' ')
    def depreOrder():
        if strlist==[]:
            return None
        rootVal=strlist.pop(0)
        if rootVal=='#':
            return None
        node=TreeNode(int(rootVal))
        node.left=depreOrder()
        node.right=depreOrder()
        return node
    root=depreOrder()
    return root
def preOder(root):
    if root ==None:
        return None
    stack=[]
    tmpNode=root
    res=[]
    while tmpNode or stack:
        while tmpNode:
            res.append(tmpNode.val)
            stack.append(tmpNode)
            tmpNode=tmpNode.left
        node=stack.pop()
        #print(node.val)
        tmpNode=node.right
    return res

class TreeNode():
    def __init__(self,x):
        self.val=x
        self.left=None
        self.right=None
if __name__=="__main__":
    t1=TreeNode(1)
    t2=TreeNode(2)
    t3=TreeNode(3)
    t4=TreeNode(4)
    t5=TreeNode(5)
    t6=TreeNode(6)
    t7=TreeNode(7)
    t8=TreeNode(8)

    t1.left=t2
    t1.right=t3
    t2.left=t4
    t2.right=t5
    t3.left=t6
    t3.right=t7
    t6.left=t8
    res=serialize(t1)
    print('序列化：',res)
    res0=deserialize(res)
    res1=preOder(res0)
    print('反序列化：',res1)


    