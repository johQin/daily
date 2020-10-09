'''
包含min函数的栈 
定义栈的数据结构，请在该类型中实现一个能够得到栈最小元素的min函数。
要求时间复杂度为O(1),
启示：时间和空间是相互代换的，用空间换时间，用时间换空间
'''
class StackWithMin():
    def __init__(self):
        self.stack=[]
        self.minstack=[]
    def pop(self):
        if self.stack==[]:
            return None
        self.minstack.pop()
        return self.stack.pop()
    def push(self,node):
        self.stack.append(node)
        if self.minstack:
            if self.minstack[-1]>node:
                self.minstack.append(node)
            else:
                self.minstack.append(self.minstack[-1])
        else:
            self.minstack.append(node)

    def minValue(self):
        if self.minstack:
            return self.minstack[-1]
        else:
            return None
        
    def top(self):
        if self.stack:
            return self.stack[-1]
        else:
            return None
if __name__ =='__main__':
    swm=StackWithMin()
    swm.push(5)
    swm.push(2)
    swm.push(3)
    swm.push(1)
    print('当前栈：{0}，当前最小值栈：{1}，当前栈内最小值：{2}，栈顶元素：{3}'.format(swm.stack,swm.minstack,swm.minValue(),swm.top()))
    swm.pop()
    swm.pop()
    print('当前栈：{0}，当前最小值栈：{1}，当前栈内最小值：{2}，栈顶元素：{3}'.format(swm.stack,swm.minstack,swm.minValue(),swm.top()))
    swm.pop()
    swm.pop()
    print('当前栈：{0}，当前最小值栈：{1}，当前栈内最小值：{2}，栈顶元素：{3}'.format(swm.stack,swm.minstack,swm.minValue(),swm.top()))

