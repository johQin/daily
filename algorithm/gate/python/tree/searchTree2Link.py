'''
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表
要求：不能创建任何新的结点，只能调整树中，结点指针的指向。
思路：

'''
def SearchTree2Link(root):
    if root==None:
        return None

    leftNode=SearchTree2Link(root.left)
    rightNode=SearchTree2Link(root.right)
    backNode=leftNode

    if leftNode:
        leftNode=findleft(leftNode)
    else:
        backNode=root
    
    root.left=leftNode
    root.right=rightNode
    if leftNode!=None:
        leftNode.right=root
    if rightNode!=None:
        rightNode.left=root
    return backNode

def findleft(root):
    node=root
    while node.right:
        node=node.right
    return node
if __name__=="__main__":
    t1=TreeNode(10)
    t2=TreeNode(8)
    t3=TreeNode(12)
    t4=TreeNode(7)
    t5=TreeNode(9)
    t6=TreeNode(11)
    t7=TreeNode(13)
    t1.left=t2
    t1.right=t3
    t2.left=t4
    t2.right=t5
    t3.left=t6
    t3.right=t7