'''
给定一棵二叉搜索树，请找出其中的第k小的结点
思路：
1.二叉搜索树的中序遍历序列是一个从小到大排列的数组，
2.故找出第k小的结点，就是取中序遍历序列数组的第k个。
'''
def SearchTreeKmin(root,k):
    if k<1:
        return None
    if k>1 and root==None:
        return None
    midlist=midOrder(root)
    print('二叉搜索树的中序序列：',midlist)
    if k>len(midlist):
        return None
    return midlist[k-1]
def midOrderRecurive(root):
    if root==None:
        return None
    midOrderRecurive(root.left)
    print(root.val)
    midOrderRecurive(root.right)
def midOrder(root):
    res=[]
    stack=[]
    tmpNode=root
    while stack or tmpNode:
        while tmpNode:
            stack.append(tmpNode)
            tmpNode=tmpNode.left
        tmpNode=stack.pop()
        res.append(tmpNode.val)
        tmpNode=tmpNode.right
    return res
class TreeNode():
    def __init__(self,x):
        self.val=x
        self.left=None
        self.right=None
if __name__=="__main__":
    t1=TreeNode(10)
    t2=TreeNode(6)
    t3=TreeNode(15)
    t4=TreeNode(4)
    t5=TreeNode(8)
    t6=TreeNode(12)
    t7=TreeNode(18)
    t8=TreeNode(11)

    t1.left=t2
    t1.right=t3
    t2.left=t4
    t2.right=t5
    t3.left=t6
    t3.right=t7
    t6.left=t8
    k=6
    kmin=SearchTreeKmin(t1,k)
    print('第%d小的数值为%d' % (k,kmin))