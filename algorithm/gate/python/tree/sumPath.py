'''
输入一棵二叉树和一个整数，打印出二叉树中结点值得和为输入整数的所有路径。
路径定义为从数的根节点开始往下一直到叶节点所经过的结点形成的一条路径了。
（注意：在返回值得list中，数组长度大的数组靠前）
思路：
1.通过广度优先遍历，将每个路径的上的结点加入到newTmpArrList上，
2.然后把同层的所有路径加到queueArrList上，
3.到了叶子结点时，计算queueArrList的元素tmpArrList上结点的和是否等于目标值，如果等于则添加到数组的第一个，因为路径长的放前面。
'''
import copy
def achieve(root,summary):
    queuelevel=[root]
    queueArrList=[[root.val]]
    res=[]
    while queuelevel:
        tmpNode=queuelevel.pop(0)
        tmpArrList=queueArrList.pop(0)
        if tmpNode.left==None and tmpNode.right==None:
            s=0
            for i in tmpArrList:
                s+=i
            if s==summary:
                res.insert(0,tmpArrList)
        if tmpNode.left:
            queuelevel.append(tmpNode.left)
            newTmpArrList=copy.copy(tmpArrList)
            newTmpArrList.append(tmpNode.left.val)
            queueArrList.append(newTmpArrList)
        if tmpNode.right:
            queuelevel.append(tmpNode.right)
            newTmpArrList=copy.copy(tmpArrList)
            newTmpArrList.append(tmpNode.right.val)
            queueArrList.append(newTmpArrList)
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
    res=achieve(t1,8)
    for i in res:
        print(i)