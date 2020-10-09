'''
遍历方式：深度优先和广度优先

深度优先
（这里指的序是相对于根节点而言的，对于左右子树来说，都是先左后右）
1.先序遍历：先访问根，再访问左子树，再访问右子树
2.中序遍历：先访问的左子树，再访问根，再访问右子树
3.后序遍历：先访问左子树，再访问右子树，最后访问根
'''

'''递归形式遍历'''
def preOrderRecurive(root):
    if root==None:
        return None
    print(root.val)
    preOrder(root.left)
    preOrder(root.right)
def midOrderRecurive(root):
    if root==None:
        return None
    midOrder(root.left)
    print(root.val)
    midOrder(root.right)
def latOrderRecurive(root):
    if root==None:
        return None
    latOrder(root.left)
    latOrder(root.right)
    print(root.val)
'''递归是可以和循环相互转换的'''

'''栈在递归中的应用'''
def preOder(root):
    if root ==None:
        return None
    stack=[]
    tmpNode=root
    while tmpNode or stack:
        while tmpNode:
            print(tmpNode.val)
            stack.append(tmpNode)
            tmpNode=tmpNode.left
        node=stack.pop()
        #print(node.val)
        tmpNode=node.right
def midOder(root):
    if root ==None:
        return None
    stack=[]
    tmpNode=root
    while tmpNode or stack:
        while tmpNode:
            stack.append(tmpNode)
            tmpNode=tmpNode.left
        node=stack.pop()
        print(node.val)
        tmpNode=node.right
def latOder(root):
    if root ==None:
        return None
    stack=[]
    tmpNode=root
    while tmpNode or stack:
        while tmpNode:
            stack.append(tmpNode)
            tmpNode=tmpNode.left
        node=stack[-1]
        
        tmpNode=node.right
        if node.right==None:
            print(node.val)
            node=stack.pop()
            while stack and node==stack[-1].right:
                node=stack.pop()
                print(node.val)
#广度优先遍历
def levelOrder(root):
    queue=[]
    tmpNode=root
    queue.append(tmpNode)
    while queue:
        tmpNode=queue.pop(0)
        print(tmpNode.val)
        if tmpNode.left:
            queue.append(tmpNode.left)
        if tmpNode.right:
            queue.append(tmpNode.right)



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
    levelOrder(t1)
'''
先序：1,2,4,5,3,6,8,7
中序：4,2,5,1,6,8,3,7
后序：4,5,2,8,6,7,3,1
'''
    
