'''
给定一个二叉树和其中一个结点，请找出此结点的下一个结点并且返回。
注意：树中的结点不仅包含左右子结点的指针，同时包含指向父结点的指针parent
思路：
1.如果这个结点有右子树，那么此结点的中序遍历顺序的下一个结点为右子树最左边的结点
2.如果这个结点没有右子树，那么此结点的中序遍历顺序的下一个结点为，直到此结点处在父结点的左树为止，返回父结点，否则为None
'''
def findMidOrderNextNode(node):
    if node.right:
        tmpNode=node.right
        while tmpNode.left:
            tmpNode=tmpNode.left
        return tmpNode
    else:
        tmpNode=node
        while tmpNode.next:
            if tmpNode.next.left==tmpNode:
                return tmpNode.next
            tmpNode=tmpNode.next
        return None


    