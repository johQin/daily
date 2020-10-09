'''
用两个栈来实现一个队列，完成队列的push和Pop操作，队列中的元素为int类型
本题知识点：栈，队列
解题思路：
栈：FILO 队列：FIFO
有栈A与栈B，
当队列push时，将元素压入栈A。例如0,1,2
当队列pop时，先从栈A中，依次将元素压入栈B，再从栈B依次出栈。
1.队列push：直接将元素压入栈A
2.队列pop():压入A=[0,1,2]---从A出栈压入B--->B=[2,1,0]-->再出栈
'''
class Stack2Queue():
    def __init__(self):
        self.acceptstack=[]
        self.outputstack=[]
    def push(self,node):
        self.acceptstack.append(node)
        return node
    def pop(self):
        if self.outputstack==[]:
            while self.acceptstack:
                self.outputstack.append(self.acceptstack.pop())
        if self.outputstack!=[]:
            return self.outputstack.pop()
        else:
            return None


if __name__ =='__main__':
    sq=Stack2Queue()
    for i in range(3):
        print('入队列',sq.push(i))

    print('栈中的数',sq.acceptstack)

    for i in range(3):
        print("出队列",sq.pop())