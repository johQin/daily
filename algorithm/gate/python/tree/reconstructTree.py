'''
由遍历序列构造二叉树
原理：由二叉树的先序序列（或后序序列或层序序列）和中序序列可以唯一的确定一颗二叉树
思路：
一、先中
1.先序序列的第一个元素一定是二叉树的根节点，这个根节点将中序序列分成左右两个子序列，
    左序列就是根节点的左子树，
    右序列就是根节点的右子树。
2.根据上面从中序序列分出来的左右序列，在先序序列中找到对应的两个序列左先序列和右先序列，
    左先序列的第一个结点，为左子树的根节点——左子根节点
    右先序列的第一个节点，为右子树的根节点——右子根节点
3.然后根据左子根节点，在左序列中分成两个左-左右子序列
      根据右子根节点，在有序列中分成两个右-左右子序列
4.如此循环分解下去...
二、后中
后序序列的最后一个节点，如同先序序列的第一个节点
三、层中

'''
def Reconstruct(pre,mid):
    root=pre[0]
    rootNode=TreeNode(root)
    position=mid.index(root)#找到左右子序列的分割位置

    midleft=mid[:position]
    midright=mid[(position+1):]

    preleft=pre[1:(position+1)]
    preright=pre[(position+1):]

    leftNode=Reconstruct(preleft,midleft)
    rightNode=Reconstruct(preright,midright)

    rootNode.left=leftNode
    rootNode.right=rightNode
    
    return rootNode
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
    
    retree=Reconstruct([1,2,4,5,3,6,8,7],[4,2,5,1,6,8,3,7])
    levelOrder(retree)
'''
先序：1,2,4,5,3,6,8,7
中序：4,2,5,1,6,8,3,7
后序：4,5,2,8,6,7,3,1
'''