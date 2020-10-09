'''
判断一棵树是否为一棵对称树
'''
def JudgeMirrorTree(left,right):
    if left==None and right==None:
        return True
    elif left==None or right==None:
        return False
    else:
        if left.val!=right.val:
            return False
        res0=JudgeMirrorTree(left.left,right.right)
        res1=JudgeMirrorTree(left.right,right.left)
        return res0 and res1
def achieve(root):
    if root==None:
        return True
    return JudgeMirrorTree(root.left,root.right)
    
class TreeNode():
    def __init__(self,x):
        self.val=x
        self.left=None
        self.right=None
if __name__=="__main__":
    t1=TreeNode(1)
    t2=TreeNode(2)
    t3=TreeNode(2)
    t4=TreeNode(4)
    t5=TreeNode(5)
    t6=TreeNode(5)
    t7=TreeNode(4)
    t8=TreeNode(8)

    t1.left=t2
    t1.right=t3
    t2.left=t4
    t2.right=t5
    t3.left=t6
    #t3.right=t7
    # t6.left=t8
    res=achieve(t1)
    print('是否对称：',res)
