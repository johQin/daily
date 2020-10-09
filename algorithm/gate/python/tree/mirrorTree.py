def mirror(root):
    if root==None:return None
    root.left,root.right=root.right,root.left
    mirror(root.left)
    mirror(root.right)
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
    mirror(t1)
    print('*'*10)
    levelOrder(t1)
