'''
输入两棵二叉树A,B，判断B是不是A的子结构。（我们约定空树不是任何数的子结构）
'''
def hasSubTree(a,b):
    if a==None or b==None:
        return False
    if a.val==b.val:
        if equals(a,b):
            return True
    res=hasSubTree(a.left,b)
    if res:
        return True
    res=hasSubTree(a.right,b)
    return res
def equals(a,b):
    if b==None:return True
    if a==None:return False

    if a.val==b.val:
        return equals(a.left,b.left) and equals(a.right,b.right)
    else:
        return False
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

    ts1=TreeNode(1)
    ts2=TreeNode(2)
    ts3=TreeNode(3)
    ts1.left=ts3
    ts1.right=ts2
    print(hasSubTree(t1,ts1))


'''
先序：1,2,4,5,3,6,8,7
中序：4,2,5,1,6,8,3,7
后序：4,5,2,8,6,7,3,1
'''


    