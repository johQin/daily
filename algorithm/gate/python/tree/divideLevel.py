'''
从上到下按层打印二叉树，同层结点从左至右输出，每层输出一行
思路：借助两个队列，如同leveOrder一样，去遍历树，
'''
def divideLevel(root):
    if root==None:
        return []
    queue0=[root]
    queue1=[]
    res=[]
    while queue0 or queue1:
        tmpArr=[]
        if queue0:
            while queue0:
                tmpNode=queue0.pop(0)
                tmpArr.append(tmpNode.val)
                if tmpNode.left:
                    queue1.append(tmpNode.left)
                if tmpNode.right:
                    queue1.append(tmpNode.right)
            res.append(tmpArr)
            tmpArr=[]
        if queue1:
            while queue1:
                tmpNode=queue1.pop(0)
                tmpArr.append(tmpNode.val)
                if tmpNode.left:
                    queue0.append(tmpNode.left)
                if tmpNode.right:
                    queue0.append(tmpNode.right)
            res.append(tmpArr)
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
    res=divideLevel(t1)
    print('层次数组：',res)