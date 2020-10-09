'''
二叉排序树，又称二叉搜索树，二叉查找树。
定义：
1.若左子树不为空，则左子树的所有结点值都小于根结点的值
2.若右子树不为空，则右子树的所有结点值都大于根结点的值
3.左右子树也分别是一棵二叉排序树
二叉排序树的中序序列是一个递增的序列
题目：输入一个整数数组，判断该数组是不是某二叉排序树的后序遍历的结果（假设输入的数组任意两个数字互不相同）
思路：
1.二叉排序树的后序序列的最后一个结点，是树的根节点。它的值将树分成左右两个子树，左子树的所有结点值都小于根节点的值，右子树的所有结点值都大于根节点的值。
2.所以一旦出现右子树序列小于根节点的值，那么此序列就不是二叉排序树的的后序遍历序列
'''
def achieve(sequence):
    if sequence==[]:
        return True
    #后序序列的最后一个值是树的根节点
    rootNum=sequence[-1]
    sequence.pop()
    position=None
    for i in range(len(sequence)):
        if (position==None) and (sequence[i]>rootNum):
            position=i
        if (position!=None) and (sequence[i]<rootNum):
            return False
    leftRes=achieve(sequence[:position])
    rightRes=achieve(sequence[position:])
    return leftRes&rightRes
class TreeNode():
    def __init__(self,x):
        self.val=x
        self.left=None
        self.right=None
def latOder(root):
    latSquence=[]
    if root ==None:
        return latSquence
    stack=[]
    tmpNode=root
    while tmpNode or stack:
        while tmpNode:
            stack.append(tmpNode)
            tmpNode=tmpNode.left
        node=stack[-1]
        
        tmpNode=node.right
        if node.right==None:
            latSquence.append(node.val)
            print(node.val)

            node=stack.pop()
            while stack and node==stack[-1].right:
                node=stack.pop()
                latSquence.append(node.val)
                print(node.val)
    return latSquence
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
    test=latOder(t1)
    print(test)
    print(achieve(test))