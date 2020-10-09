'''
输入两个整数序列，第一个序列为栈的压入顺序，请判断第二个序列是否可能为栈的弹出序列
（压入和弹出可以交叉进行）
思路：
1. 借助一个辅助栈，用于压入pushs的压栈序列
2. 当每向辅助栈压入一个元素后，就去判断辅助栈栈顶元素是否等于pops的弹栈序列元素，
3. 如果相等，辅助栈弹出一个元素，并且继续比较当前辅助栈栈顶元素是否与弹栈序列的下一个相等，
    - 如相等，辅助栈继续弹栈，index+1
    - 如不相等，跳出比较的循环结构，
4. 最后判断辅助栈是否为空，如果为空，则返回true。如果不为空，则返回false
'''
class PushPopSequence():
    def achieve(self,pushs,pops):
        index=0
        stack=[]
        for item in pushs:
            stack.append(item)
            while stack and stack[-1]==pops[index]:
                stack.pop()
                index+=1
        if stack:
            return False
        else:
            return True

if __name__=='__main__':
    pps=PushPopSequence()
    flag=pps.achieve([1,2,3],[2,1,3])
    print(flag)