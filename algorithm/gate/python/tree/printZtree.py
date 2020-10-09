'''
实现一个函数按照之字形打印二叉树，
即：第一行从左至右打印，第二行从右至左打印
思路：
1.借助两个栈，一个栈stack0用于存取第一行数据，一个栈stack1用于存取第二行数据，交替存取
2. 当遍历打印stack0时，将stack0的结点的左右子树，按照先左后右的次序，往stack1中添加。保证输出为从左至右
3. 当遍历打印stack1时，将stack1的结点的左右子树，按照先右后左的次序，往stack1中添加。保证输出为从右至左
'''
def ZlevelOrder(root):
    if root==None:
        return []
    stack0=[root]
    stack1=[]
    res=[]
    while stack0 or stack1:
        if stack0:
            while stack0:
                tmpNode=stack0.pop()
                res.append(tmpNode.val)
                # print(tmpNode.val)
                if tmpNode.left:
                    stack1.append(tmpNode.left)
                if tmpNode.right:
                    stack1.append(tmpNode.right)
        if stack1:
            while stack1:
                tmpNode=stack1.pop()
                res.append(tmpNode.val)
                # print(tmpNode.val)
                if tmpNode.right:
                    stack0.append(tmpNode.right)
                if tmpNode.left:
                    stack0.append(tmpNode.left)
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
    res=ZlevelOrder(t1)
    print('z型数组：',res)
    