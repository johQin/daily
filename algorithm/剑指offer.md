# 剑指offer

基于python3.6

# 数组

## 数组基本操作

```python
if __name__=='__main__':
    tlist=[1,2,3,4]
    #都是对原数组的操作
    tlist.append(6)
    tlist.extend([7,8,9])
    tlist.insert(4,5)
    print("数组添加操作，append(obj),extend(seq),insert(index,obj),当前数组：%s" % tlist)
    tlist.pop()#等价于tlist.pop(-1)，python中pop(index)，默认index=-1，弹出list的最后一个元素
    tlist.pop(0)#弹出list的第一个元素，少了1
    tlist.pop(1)#弹出索引为1的元素，少了3
    tlist.remove(8)#remove(obj),删除第一个匹配obj的元素
    print("数组删除操作，pop(index),当前数组：%s" % tlist)

    tlist.reverse()
    print("数组反向：%s" % tlist)

    print('匹配对象的索引：',tlist.index(6))#找到第一个匹配项的索引，如果没有找到对象则抛出异常。
    print('匹配对象的出现的次数：',tlist.count(6))
```

## 二分查找法

针对有序列表，查找相应元素

```python
import math
class TwoSplitFind():
    def achieve(self,arr,target):
        flag=-1
        end=len(arr)-1
        start=0
        while start<=end :
            mid=math.floor((start+end)/2)
            '''
            或者采用整数除法'//',例如(start+end)//2
            或者采用移位操作'>>',例如(start+end)>>1
            '''
            if target<arr[mid]:
                end=mid-1
            elif target>arr[mid]:
                start=mid+1
            else:
                flag=mid
                break
        return flag
if __name__ =='__main__':
    tsf=TwoSplitFind()
    res=tsf.achieve([0,1,2,3,4,5,6,7,8,9,10],11)
    print("查找结果=",res)
```

## 有序二维数组查找

```python

'''
在一个二维数组中(每一个一维数组的长度相同)，
每一行都按照从左到右递增的顺序排序，
每一列都按照从上到下递增的顺序排序。
请完成一个函数，
输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
'''
class ArrayFind():
    '''
    此算法复杂度为m+n，在于利用了从上到下，从左到右依次增大的条件
    '''
    def find(self,array2D,target):
        flag=False
        location={'x':-1,'y':-1}
        i=0
        j=len(array2D[0])-1
        while i<len(array2D) and j >=0:
            val=array2D[i][j]
            if target==val:
                flag=True
                location['x']=i
                location['y']=j
                break
            elif target<val:
                j -= 1
            else:
                i += 1
        return {'flag':flag,'location':location}       
if __name__=='__main__':
    af=ArrayFind()
    arr=[[1,2,3],[4,5,6],[7,8,9]]
    tar=1
    res=af.find(arr,8)
    print('res：',res)

```

## 旋转数组的最小值

```python
'''
输入一个递增数组的旋转数组，找出旋转数组中的最小值
旋转数组解释如下：
数组元素依次向右移位，最末尾元素循环到数组首部
输入: [1,2,3,4,5,6,7] 和 k = 3
向右旋转 1 步: [7,1,2,3,4,5,6]
向右旋转 2 步: [6,7,1,2,3,4,5]
向右旋转 3 步: [5,6,7,1,2,3,4]
'''
class RotateArrayFind():
    def achieve(self,arr):
        start=0
        end=len(arr)-1
        while start<=end:
            mid=(start + end) //2
            if arr[mid]<arr[mid-1]:#由于原数组是一个递增数列，如果旋转数组的mid左边的数比当前mid的数大，那么mid的数就是最小值
                return arr[mid]
            elif arr[mid]<arr[end]:#当前这种情况说明旋转的步数还未过半，最小的数在mid的左侧，所以end=mid-1，让他去左边查找。
                end=mid-1
            else:#当前这种情况说明旋转的步数还已过半，最小的数在mid的右侧，
                start=mid+1
        return 0


if __name__=='__main__':
    raf=RotateArrayFind()
    res=raf.achieve([4,5,6,7,1,3])
    print('最小值：',res)

```

## 基本排序

<h5>冒泡排序

```python
'''
冒泡排序：
是用作一旦找到一个比当前大的值就交换
'''
class MaoPaoSort():
    def achieve(self,arr):
        for i in range(len(arr)):
            for j in range(len(arr)-i-1):
                if arr[j+1]<arr[j]:
                    arr[j],arr[j+1]=arr[j+1],arr[j]
if __name__ =='__main__':
    mps=MaoPaoSort()
    arr=[10,24,1,6,7,8,41,16,30,15,17]
    mps.achieve(arr)
    print('冒泡排序后的数组',arr)
```

<h5>选择法排序

```python
'''
选择法排序：
通过找到最大值的位置，放到数组的最后，交换对应位置值，然后执行下一轮查找排序
'''
class ChooseSort():
    def achieve(self,arr):
        maxp=0
        tmp=None
        for i in range(len(arr)):
            for j in range(len(arr)-i):
                if arr[maxp]<arr[j]:
                    maxp=j
            tmp=arr[len(arr)-i-1]
            arr[len(arr)-i-1]=arr[maxp]
            arr[maxp]=tmp
            maxp=0#每次找到最大值位置并交换值之后，需要将maxp置为首个元素索引0。
if __name__=='__main__':
    cs=ChooseSort()
    arr=[4,3,6,5,8,9,1]
    cs.achieve(arr)
    print('选择排序后的数组：{0}'.format(arr))
```

## 数组奇偶调整

```python
'''
题目：
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分。
并保证奇数和奇数，偶数和偶数之间的相对位置不变。

稳定型：奇数与奇数之间，偶数与偶数之间相对位置不变，简单调整和冒泡法调整都属于稳定调整。

非稳定型：相对位置发生变化，
非稳定型实现：
两个指针，p指向第一个元素，q指向最后一个元素。
两个指针遍历数组，向后移动p使得p指向偶数，向前移动q使得q指向奇数，交换p和q的内容。
指针p位于指针q后面时，结束遍历。
'''
class ArrayOddEvenAdjust():
    #时间复杂度O(n)，空间复杂度O(n)
    def sampleAchieve(self,arr):
        tmp=[]
        for i in arr:
            if i % 2==1:
                tmp.append(i)
        for i in arr:
            if i % 2==0:
                tmp.append(i)
        return tmp
    #时间复杂度O(n^2)，空间复杂度O(1)
    def maopaoAchieve(self,arr):
        tmp=None
        for i in range(len(arr)):#range(start,end,step)，
            for j in range(len(arr)-i-1):#数组倒序range(len(arr)-1,-1,-1)
                 if arr[j]%2==0 and arr[j+1]%2==1 :
                     tmp = arr[j]
                     arr[j] = arr[j+1]
                     arr[j+1] = tmp
    def unstableAchieve(self,arr):
        left=0
        right=len(arr)-1
        tmp=None
        while left<right:
            while left<right and arr[left]%2 ==1:#找偶数的索引，如果为奇数那么自增1，直到找到偶数
                left+=1 
            while left<right and arr[right]%2==0:#找奇数的索引，如果为偶数那么继续自减1，直到找到奇数
                right-=1
            if left<right:
                tmp=arr[left]
                arr[left]=arr[right]
                arr[right]=tmp
            

if __name__ =='__main__':
    aoea=ArrayOddEvenAdjust()
    inputarr=[7,8,4,5,2,6,-1,-3,9,13]
    res=aoea.sampleAchieve(inputarr)
    print("简单奇偶调整：原数组：{0} 调整后数组：{1}".format(inputarr,res))
    inputarr1=[7,8,4,5,2,6,-1,-3,9,13]
    aoea.maopaoAchieve(inputarr1)#传址
    print("冒泡法奇偶调整：原数组：{0} 调整后数组：{1}".format(inputarr,inputarr1))
    inputarr2=[7,8,4,5,2,6,-1,-3,9,13]
    aoea.unstableAchieve(inputarr2)
    print("不稳定法奇偶调整：原数组：{0} 调整后数组：{1}".format(inputarr,inputarr2))
```

## 最大子序列和

```python
'''
给一个数组，求最大的子序列和。
思路：
1.maxSum用于存储历史最大的子序列和，tmpSum用于存储临时子序列和，
2.如果tmpSum+item<item的话，那么子序列还不如从当前位置开始子序列，reStart为可能的最大子序列和的起点。
3.否则，子序列依旧向后延伸
3.当前的tmpSum的子序列和大于历史最大子序列和时，那么此时的index为终止结点，reStart为最大子序列的起点
'''
def MaxSubArrSum(arr):
    maxSum=None
    tmpSum=0
    index=0
    pos0=0
    pos1=0
    reStart=0
    for item in arr:
        if maxSum==None:
            maxSum=item
            pos0=index
            pos1=index
        if tmpSum+item<item:
            tmpSum=item
            reStart=index
        else:
            tmpSum+=item
        if maxSum<tmpSum:
            maxSum=tmpSum
            pos0=reStart
            pos1=index
        index+=1
    return maxSum,pos0,pos1
if __name__=='__main__':
    arr=[6,-3,-2,7,-15,1,2,2]
    maxSum,pos0,pos1=MaxSubArrSum(arr)
    print('最大子序列和',maxSum,pos0,pos1)

```

# 斐波那契数列

```python
'''
斐波那契数列
递推公式
如果在使用递归的时候涉及到重复计算，
如果“前面'计算'出的结果”对后面的计算有影响，那么我们通常会保留此结果，以供给下一次计算使用。
增加临时变量，存储结果
'''
class Resolve():
    def fibonacci(self,n):
        if n==0 :
            return 0
        elif n==1 :
            return 1
        elif n>1:
            a=0
            b=1
            c=0
            for i in range(0,n-1):
                c=a+b
                a=b
                b=c
            return c
        else:
            return "param is invalid"
    def complexFibonacci(self,n):
        if n==0:
            return 0
        elif n==1:
            return 1
        elif n>1:
            return self.complexFibonacci(n-1)+self.complexFibonacci(n-2)
        else:
            return "param is invalid"
if __name__ =='__main__' :
    fb=Resolve()
    n=4
    res=fb.fibonacci(n)
    print("正确的算法之斐波那契数列的第%d个数是%d,算法复杂度为O(n)"% (n,res)) 
    res2=fb.complexFibonacci(n)
    print("错误的算法之斐波那契数列的第%d个数是%d,算法复杂度为O(2的n次方)"%(n,res2))
"""
python的占位符操作：https://www.cnblogs.com/a389678070/p/9559360.html
常见的占位符有：
%d    整数
%f    浮点数
%s    字符串
%x    十六进制整数
tpl = "i am %s" % "alex"
tpl = "i am %s age %d" % ("alex", 18)
注意：
1.模板字符串和参数之间以 % 号相隔
2.多个参数用括号括起来
3.还有format用法
"""
    
```

## 青蛙跳台阶

```python
'''
递推公式
青蛙跳台阶
题目：一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个n级的台阶总共有多少种跳法？(先后词序不同也算不同的结果)
答题思路
如果只有1级台阶，那显然只有一种跳法
如果有2级台阶，那么就有2种跳法，一种是分2次跳。每次跳1级，另一种就是一次跳2级
如果台阶级数大于2，设为n的话，这时我们把n级台阶时的跳法看成n的函数，记为,第一次跳的时候有2种不同的选择：
一是第一次跳一级，此时跳法的数目等于后面剩下的n-1级台阶的跳法数目，即为f(n-1)
二是第一次跳二级，此时跳法的数目等于后面剩下的n-2级台阶的跳法数目，即为f(n-2),
因此n级台阶的不同跳法的总数为f(n-1)+f(n-2)，不难看出就是斐波那契数列

变式：若把条件修改成一次可以跳一级，也可以跳2级...也可以跳上n级呢？
f(n)=f(n-1)+f(n-2)+f(n-3)+....+f(2)+f(1)
f(n-1)=f(n-2)+f(n-3)+....+f(2)+f(1)
两式相减得：
f(n)=2f(n-1)
'''
class Frog():
    def jump2Plans(self,n):
        if n==1:
            return 1
        elif n==2:
            return 2
        elif n>2:
            a=1
            b=2
            for i in range(n-1):
                c=a+b
                a=b
                b=c
            return c
        else:
            return 0
    def jumpNPlans(self,n):
        if n==1:
            return 1
        elif n>1:
            a=1
            for i in range(n-1):
                b=2*a
                a=b
            return b
        else:
            return 0
        
if __name__=='__main__':
    f=Frog()
    n=5;
    res=f.jump2Plans(n)
    res1=f.jumpNPlans(n)
    print("青蛙一次可以跳一级也可以跳两级，一共有%d级台阶，共有%d种跳跃方案" % (n,res))
    print("青蛙一次可以跳1,2,..n级，一共有%d级台阶，共有%d种跳跃方案" % (n,res1))
```

## 约瑟夫环

```python
'''
题目描述：
每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。
HF作为牛客的资深元老,自然也准备了一些小游戏。
其中,有个游戏是这样的:
首先,让小朋友们围成一个大圈。
然后,他随机指定一个数m,让编号为0的小朋友开始报数。
每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,
从他的下一个小朋友开始,继续0…m-1报数…这样下去…
直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版
请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)
思路：https://blog.csdn.net/qq_37886086/article/details/89705044
约瑟夫环的问题

递推公式：
f(1)=0
f(i)=[f(i-1)+m]%n
'''
def maxWinner(n,m):
    if n<1 or m<1:
        return None
    if n==1:
        return 0
    pre=0
    cur=0
    for curN in range(2,n):
        cur=(pre+m)%curN
        pre=cur
        print(cur,pre)
    return pre
if __name__=="__main__":
    print("最大的赢家编号：%s" % maxWinner(10,9))
```

## 以小填大

```python
'''
用n个2*1的长方形填充2*n长方形，一共有多少种填充法
思路：
1.当竖着放最后一个的时候，还有2*(n-1)的长方形需要填充
2.当横着放最后一个的时候，倒数第二个只能横着放，但还有2*(n-2)的长方形需要填充，
3.故f(n)=f(n-1)+f(n-2)
4.当n=2时，f(2)=2
5.当n=1时，f(1)=1
'''
def smallFillbig(n):
    if n<=0:
        return 0
    elif n==1:
        return 1
    elif n==2:
        return 2
    else:
        a=1
        b=2
        for i in range(2,n):
            c=a+b
            a=b
            b=c
        return c
if __name__=='__main__':
    res=smallFillbig(3)
    print('hah',res)
```

# 栈

## 压栈和出栈序列

```python
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
```

## 栈实现队列

```python
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
```

## 包含min函数的栈

```python
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


```

# 链表

```python

#单向链表结点
class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
def printLinkList(node):
    while node:
        print(node.val)
        node=node.next

if __name__=='__main__':
    l1=SingleDirectionListNode(1)
    l2=SingleDirectionListNode(2)
    l3=SingleDirectionListNode(3)
    l1.next=l2
    l2.next=l3
    printLinkList(l1)
```



## 链表的公共结点

```python

'''
输入两个链表，找出它们第一个公共结点。
第一个公共结点之后，两个链表的结点就相同了
思路：
1.首先借助两个指针分别指向两个链表
2.找出它们的链表长度差k，然后让长的那个链表的指针先走到第k个位置。
3.由于第一个公共结点之后，两个链表的结点就相同了。
    从链表末尾的角度来看，较长的链表的指针处在结点k的位置上，较短的那个链表的指针处在开头。
    如果同时后移这两个指针，他们两个会同时在第一个公共结点处相遇，
    如果到链表的末尾处，这两个指针都没有指向同一个结点，那么便不存在公共的结点
'''
class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
def findLongerk(link1,link2):
    l1=link1
    l2=link2
    while l1 and l2:#一旦有一个链表的指针走到末尾，那么就跳出
        l1=l1.next
        l2=l2.next
    k=0
    #判断哪一个链表较长
    if l1:#链表较长的那一个继续向后移动，并计数，直到移到链表末尾，那么k就为两个链表的长度之差
        while l1:
           k+=1
        return find1thComNode(link1,link2,k)
    else:
        while l2:
            k+=1
        return find1thComNode(link2,link1,k)
def find1thComNode(long,short,k):
    for i in range(k):#让长的那个指针先向后移k步
        long=long.next
    while long!=short:#一旦两个指针同时指向同一个结点（第一个公共结点），就跳出循环。
        long=long.next
        short=short.next
    return long#返回指针所指向的结点



if __name__=='__main__':
    l1=SingleDirectionListNode(1)
    l2=SingleDirectionListNode(2)
    l3=SingleDirectionListNode(3)
    l4=SingleDirectionListNode(4)
    l5=SingleDirectionListNode(5)
    l6=SingleDirectionListNode(6)
    l7=SingleDirectionListNode(7)
    l1.next=l2
    l2.next=l5
    l3.next=l4
    l4.next=l5
    l5.next=l6
    l6.next=l7
    com=findLongerk(l1,l3)
    print('公共结点值%s' % com.val if com else None)
```

## 单增链表合并

```python
'''
输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则
思路：
1. 还是先判断链表是否为空的情况
2. 从两个链表的头的val来看，谁小谁就作为函数返回的头head
3. 在借助三个指针prev，h1，h2
    prev:作为部分已经合并后的链表的尾部
    h1：link1未合并部分的头部
    h2：link2未合并部分的头部
    通过比较h1与h2的val谁小，prev的next就指向谁，
    然后prev=值小的头部
    被合并的链表头部向未被合并的部分移动
4.循环，直到h1或h2有一个为None，
5.prev.next指向未等于None的头部
6.最后返回head
'''
class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
def Merge(link1,link2):
    if link1==None and link2 ==None:
        return None
    elif link1==None:
        return link2
    elif link2==None:
        return link1
    else:
        pass
    head=link1 if link1.val<link2.val else link2#做输出用
    h1=link1
    h2=link2
    prev=head
    if h1==head:
        h1=head.next
    else:
        h2=head.next
    while h1 and h2:
        if h1.val<h2.val:
            prev.next=h1
            prev=h1
            h1=h1.next
        else:
            prev.next=h2
            prev=h2
            h2=h2.next
    if h1==None:
        prev.next=h2
    else:
        prev.next=h1
    
    return head
    
if __name__=="__main__":
    l1=SingleDirectionListNode(2)
    l2=SingleDirectionListNode(2)
    l3=SingleDirectionListNode(3)
    l4=SingleDirectionListNode(4)
    l5=SingleDirectionListNode(5)
    l6=SingleDirectionListNode(6)
    l1.next=l2
    l2.next=l3
    l4.next=l5
    l5.next=l6
    print(Merge(l1,l4).val)
    
```

## 复杂链表克隆

```python
'''
输入一个复杂链表，链表的每一个结点包含val，next（指向下一个结点），random（指向一个任意的结点）
克隆这个复杂链表，返回原链表的副本。
思路：
1.复制每一个结点形成副本，并且将副本紧插入到原结点的后面，暂时不做random的指向工作
2.实现新建node的random指向
3.断开副本与原本的链接
'''

class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
        self.random=None
def Clone(startNode):
    
    #复制每一个结点形成副本，并且将副本紧插入到原结点的后面，暂时不做random的指向工作
    tmp=startNode
    while tmp:
        node=SingleDirectionListNode(tmp.val)
        node.next=tmp.next
        tmp.next=node
        tmp=node.next
    
    #实现新建node的random指向
    tmp=startNode
    while tmp:
        if tmp.random:
            tmp.next.random=tmp.random.next
        #tmp.next作为tmp的副本，tmp.next.ramdom是副本的random
        #tmp作为原本，tmp.random作为原本的random，tmp.random.next作为原本的random的副本
        #将原本的random的副本赋值给原本的副本的random
        #这解决的副本random的指向问题
        tmp=tmp.next.next
    
    #断开副本与原本的链接
    tmp=startNode
    tmpr=startNode.next
    res=startNode.next
    while tmpr:
        tmp.next=tmpr.next
        tmp=tmpr
        tmpr=tmpr.next
    
    return res

    
if __name__=="__main__":
    l1=SingleDirectionListNode(1)
    l2=SingleDirectionListNode(2)
    l3=SingleDirectionListNode(3)
    l4=SingleDirectionListNode(4)
    l5=SingleDirectionListNode(5)
    l6=SingleDirectionListNode(6)
    l1.next=l2
    l1.random=l3
    l2.next=l3
    l2.random=l4
    l3.next=l4
    l3.random=l5
    l4.next=l5
    l4.random=l6
    l5.next=l6
    l5.random=l3
    l6.random=None
    res=Clone(l1)
    while res:
        print('结点值：{0},结点的随机值:{1}'.format(res.val,res.random.val if res.random else None))
        res=res.next
```



## 从尾到头返回列表

```python
'''
输入一个链表，按链表值从尾到头的顺序返回一个数组
'''
#单向链表结点
class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
    def toString(self):
        return '(val:{0},next:{1})'.format(self.val,self.next)
def LinkListT2H(startNode):
    if not isinstance(startNode,SingleDirectionListNode):
        return None
    target=[]
    node=startNode
    while node:
        target.append(node.val)
        node=node.next
    target.reverse()
    return target 
if __name__=="__main__":
    l1=SingleDirectionListNode(1)
    l2=SingleDirectionListNode(2)
    l3=SingleDirectionListNode(3)
    l1.next=l2
    l2.next=l3
    print('从尾到头的顺序返回一个数组',LinkListT2H(l1))

```

## 是否含有环

```python
'''
给一个链表若其中包含环，请找到该链表环的入口结点，否则返回None
思路：
1.需要定义两个指针（快慢指针），慢指针一次向后移动一个结点，快指针一次向后移动两个结点，
2.如果链表无环，快指针会先遇到None，
3.如果链表有环，快指针会与慢指针相遇。
    假设有环的链表的结构为：0--s-->ce--m-->meet
                                 ↑<---re----↓
    fast走的距离为2x,slow走的距离为x，在环的meet结点相遇。
    ce——circle entry
    s——straight path
有：
    x=s+m
    2x=n(m+re)+s+m
得：
    s+m=n(m+re)
终：
    s=re+(n-1)(m+re)
从上面这个式子可得
如果有一个指针从链表的头部出发，一个指针从fast与slow两个点相遇的结点出发，他们会同时到达环的入口处
                                  
                                  
'''
class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
def isWithCircle(head):
    #快慢指针用来确定是否存在环
    fast=head
    slow=head
    while fast and fast.next:
        slow=slow.next
        fast=fast.next.next
        print('fast:%d,slow:%d' % (fast.val,slow.val))
        if fast==slow:
            break
    if  not (fast and fast.next):
        return None
    #确定环的入口
    tmp=head
    while tmp!=fast:
        tmp=tmp.next
        fast=fast.next
    return tmp
if __name__=="__main__":
    l1=SingleDirectionListNode(1)
    l2=SingleDirectionListNode(2)
    l3=SingleDirectionListNode(3)
    l4=SingleDirectionListNode(4)
    l5=SingleDirectionListNode(5)
    l6=SingleDirectionListNode(6)
    l7=SingleDirectionListNode(7)
    l1.next=l2
    l2.next=l3
    l3.next=l4
    l4.next=l5
    l5.next=l6
    l6.next=l7
    l7.next=l4
    print('是否有环，环的位置在%s' % (isWithCircle(l1).val if isWithCircle(l1) else None))

```

## 链表反转

```python
'''
输入一个链表，反转链表后，输出新链表的表头
思路：
1.当链表结点只有一个时，直接返回该结点
2.当链表有多个结点时，采用三个指针left，node，right
3.将三个指针分别初始化，left=startNode,node=start.next,right=node.next,并将首结点的next赋值为None
4.然后将node.next指向left，node与right之间的链接就断掉了，此后依次将left、node、right往后移
5.直到right为空，跳出循环，跳出之后，需要将node.next=left
'''
class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
def reverse(startNode):
    node=startNode
    if node==None:
        return None
    if node.next==None:
        return node
    left=node
    node=node.next
    right=node.next
    left.next=None
    while right!=None:
        node.next=left
        left=node
        node=right
        right=right.next
    node.next=left
    return node



if __name__=="__main__":
    l1=SingleDirectionListNode(1)
    l2=SingleDirectionListNode(2)
    l3=SingleDirectionListNode(3)
    l4=SingleDirectionListNode(4)
    l5=SingleDirectionListNode(5)
    l6=SingleDirectionListNode(6)
    l1.next=l2
    l2.next=l3
    l3.next=l4
    l4.next=l5
    l5.next=l6
    print(reverse(l1).val)
```

## 倒数第k个结点

```python
'''
输入一个链表，输出该链表中倒数第k个节点
思路：
1.借助两个指针left和right，初始化两个指针为头结点
2.然后让两个指针，间隔k个距离，如果链表比k短，那么直接返回None
3.然后让间隔k个距离的两个指针，同时向后移动，直到right指针为None，返回left

'''
class SingleDirectionListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
def TailKthNode(startNode,k):
    if not isinstance(startNode,SingleDirectionListNode):
        return None
    left=startNode
    right=startNode
    for i in range(k):
        if not right:
            return None
        right=right.next
    while right!=None:
        right=right.next
        left=left.next
    return left.val

if __name__=="__main__":
    l1=SingleDirectionListNode(1)
    l2=SingleDirectionListNode(2)
    l3=SingleDirectionListNode(3)
    l4=SingleDirectionListNode(4)
    l5=SingleDirectionListNode(5)
    l6=SingleDirectionListNode(6)
    l1.next=l2
    l2.next=l3
    l3.next=l4
    l4.next=l5
    l5.next=l6
    print(TailKthNode(l1,3))
```

# 树

```python
class TreeNode():
    def __init__(self,x):
        self.val=x
        self.left=None
        self.right=None
```

## 树的遍历

```python
'''
遍历方式：深度优先和广度优先

深度优先
（这里指的序是相对于根节点而言的，对于左右子树来说，都是先左后右）
1.先序遍历：先访问根，再访问左子树，再访问右子树
2.中序遍历：先访问的左子树，再访问根，再访问右子树
3.后序遍历：先访问左子树，再访问右子树，最后访问根
'''

'''递归形式遍历'''
def preOrderRecurive(root):
    if root==None:
        return None
    print(root.val)
    preOrder(root.left)
    preOrder(root.right)
def midOrderRecurive(root):
    if root==None:
        return None
    midOrder(root.left)
    print(root.val)
    midOrder(root.right)
def latOrderRecurive(root):
    if root==None:
        return None
    latOrder(root.left)
    latOrder(root.right)
    print(root.val)
'''递归是可以和循环相互转换的'''

'''栈在递归中的应用'''
def preOder(root):
    if root ==None:
        return None
    stack=[]
    tmpNode=root
    while tmpNode or stack:
        while tmpNode:
            print(tmpNode.val)
            stack.append(tmpNode)
            tmpNode=tmpNode.left
        node=stack.pop()
        #print(node.val)
        tmpNode=node.right
def midOder(root):
    if root ==None:
        return None
    stack=[]
    tmpNode=root
    while tmpNode or stack:
        while tmpNode:
            stack.append(tmpNode)
            tmpNode=tmpNode.left
        node=stack.pop()
        print(node.val)
        tmpNode=node.right
def latOder(root):
    if root ==None:
        return None
    stack=[]
    tmpNode=root
    while tmpNode or stack:
        while tmpNode:
            stack.append(tmpNode)
            tmpNode=tmpNode.left
        node=stack[-1]
        
        tmpNode=node.right
        if node.right==None:
            print(node.val)
            node=stack.pop()
            while stack and node==stack[-1].right:
                node=stack.pop()
                print(node.val)
#广度优先遍历
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
'''
先序：1,2,4,5,3,6,8,7
中序：4,2,5,1,6,8,3,7
后序：4,5,2,8,6,7,3,1
'''
    

```

## 由遍历序列构造二叉树

```python
'''
由遍历序列构造二叉树
原理：由二叉树的先序序列（或后序序列或层序序列）和中序序列可以唯一的确定一颗二叉树
思路：
一、先中
1.先序序列的第一个元素一定是二叉树的根节点，这个根节点将中序序列分成左右两个子序列，
    左序列就是根节点的左子树，
    右序列就是根节点的右子树。
2.根据上面从中序序列分出来的左右序列，在先序序列中找到对应的两个序列左先序列和右先序列，
    左先序列的第一个结点，为左子树的根节点——左子根节点
    右先序列的第一个节点，为右子树的根节点——右子根节点
3.然后根据左子根节点，在左序列中分成两个左-左右子序列
      根据右子根节点，在有序列中分成两个右-左右子序列
4.如此循环分解下去...
二、后中
后序序列的最后一个节点，如同先序序列的第一个节点
三、层中

'''
def Reconstruct(pre,mid):
    root=pre[0]
    rootNode=TreeNode(root)
    position=mid.index(root)#找到左右子序列的分割位置

    midleft=mid[:position]
    midright=mid[(position+1):]

    preleft=pre[1:(position+1)]
    preright=pre[(position+1):]

    leftNode=Reconstruct(preleft,midleft)
    rightNode=Reconstruct(preright,midright)

    rootNode.left=leftNode
    rootNode.right=rightNode
    
    return rootNode
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
    
    retree=Reconstruct([1,2,4,5,3,6,8,7],[4,2,5,1,6,8,3,7])
    levelOrder(retree)
'''
先序：1,2,4,5,3,6,8,7
中序：4,2,5,1,6,8,3,7
后序：4,5,2,8,6,7,3,1
'''
```

## 最小k个数

```python
'''
输入n个数，找出其中最小的k个数
思路：
1.如果采用排序做这个，算法复杂度是O(n^2)
2.如果用只有k个元素的数组，并且k个元素按照从小到大的顺序排好（用折半查找法去查询下一个数该放什么位置，复杂度为O( log(n) )）
    在查找到后，还要进行数字的顺序移动复杂度为O(n)，那么整个算法的复杂度为O(nlog(n))
3.树的最大堆和最小堆的方法：
最大堆：一个完全二叉树，根节点的值大于左右子树的关键值。
最小堆：一个完全二叉树，根节点的值小于左右子树的关键值。
这一堆有k个结点，查找算法复杂度为O(log(n)),替换值的算法复杂度为O(log(n))，故整个算法复杂度为O(log(n)*log(n))，

最大堆用于查找最小的k个数，如果一个数比根节点的值小，那么拿出根节点的值，放入这个数进堆，然后再与左右两个子树的比较，放到适合的位置
最小堆用于查找最大的k的数，如果一个数比根节点的值大，那么拿出根节点的值，放入这个数进堆，。。。。

含有n个结点的完全二叉树用数组表示，父结点与左右子结点的索引关系
父结点的索引为i，当2i<=n时，左子结点的索引为2i+1，右子结点的索引为2i+2
子节点的索引为i，父结点的索引为(i-1)整除2 

采用最大堆解决，查找最小的k个数问题
这个过程分两步，
1.创建含有k个结点的初始最大堆，从数组末尾开始创建。
2.与后n-k个值做比较，调整最大堆。


'''
def initCreatMaxHeap(karr):
    maxHeap=[]
    for item in karr:
        maxHeap.append(item)
        curIndex=len(maxHeap)-1
        while curIndex!=0:
            parentIndex=(curIndex-1)>>1
            if maxHeap[parentIndex]<maxHeap[curIndex]:
                maxHeap[curIndex],maxHeap[parentIndex]=maxHeap[parentIndex],maxHeap[curIndex]
                curIndex=parentIndex
            else:
                break
    return maxHeap
def AdjustMaxHeap(maxHeap,num):
    curIndex=0
    length=len(maxHeap)
    if num<maxHeap[0]:
        maxHeap[0]=num
        while curIndex<length:
            leftChildIndex=2*curIndex+1
            rightChildIndex=2*curIndex+2
            largerIndex=0
            if rightChildIndex<length:
                if maxHeap[rightChildIndex]<maxHeap[leftChildIndex]:
                    largerIndex=leftChildIndex
                else:
                    largerIndex=rightChildIndex
            elif leftChildIndex<length:
                largerIndex=leftChildIndex
            else:
                break
            if maxHeap[curIndex]<maxHeap[largerIndex]:
                maxHeap[curIndex],maxHeap[largerIndex]=maxHeap[largerIndex],maxHeap[curIndex]
                curIndex=largerIndex
            else:
                break
def achieve(arr,k):
    if len(arr)<k or k<=0:
        return []
    maxHeap=[]
    i=0
    length=len(arr)
    while i<k:
        maxHeap.append(arr[i])
        i+=1
    maxHeap=initCreatMaxHeap(maxHeap)
    while i<length:
        AdjustMaxHeap(maxHeap,arr[i])
        print('最大堆',maxHeap)
        i+=1

    return maxHeap
if __name__=='__main__':
    arr=[1,7,9,10,25,14,16,98,54,27,5,4,45,36,12]#
    k=10
    achieve(arr,k)

    

```

## 子树

```python
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


    
```

## 树的对称性

```python
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

```

## 结点的下一个结点

```python
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
```

## 树的镜像

```python
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

```

## 按z形打印树

```python
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
    
```

## 树的层次数组

```python
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
```

## 数组是不是二叉排序树后序遍历序列

```python
'''
二叉排序树，又称二叉搜索树，二叉查找树。
定义：
1.若左子树不为空，则左子树的所有结点值都小于根结点的值
2.若右子树不为空，则右子树的所有结点值都大于根结点的值
3.左右子树也分别是一棵二叉排序树
二叉排序树的中序序列是一个递增的序列
题目：输入一个整数数组，判断该数组是不是某二叉排序树的后序遍历的结果（假设输入的数组任意两个数字互不相同）
思路：
1.二叉排序树的后序序列的最后一个结点，是树的根节点。它的值将树分成左右两个子树，左子树的所有结点值都小于根节点的值，右子树的所有结点值都大于根节点的值。
2.所以一旦出现右子树序列小于根节点的值，那么此序列就不是二叉排序树的的后序遍历序列
'''
def achieve(sequence):
    if sequence==[]:
        return True
    #后序序列的最后一个值是树的根节点
    rootNum=sequence[-1]
    sequence.pop()
    position=None
    for i in range(len(sequence)):
        if (position==None) and (sequence[i]>rootNum):
            position=i
        if (position!=None) and (sequence[i]<rootNum):
            return False
    leftRes=achieve(sequence[:position])
    rightRes=achieve(sequence[position:])
    return leftRes&rightRes
class TreeNode():
    def __init__(self,x):
        self.val=x
        self.left=None
        self.right=None
def latOder(root):
    latSquence=[]
    if root ==None:
        return latSquence
    stack=[]
    tmpNode=root
    while tmpNode or stack:
        while tmpNode:
            stack.append(tmpNode)
            tmpNode=tmpNode.left
        node=stack[-1]
        
        tmpNode=node.right
        if node.right==None:
            latSquence.append(node.val)
            print(node.val)

            node=stack.pop()
            while stack and node==stack[-1].right:
                node=stack.pop()
                latSquence.append(node.val)
                print(node.val)
    return latSquence
if __name__=="__main__":
    t1=TreeNode(10)
    t2=TreeNode(8)
    t3=TreeNode(12)
    t4=TreeNode(7)
    t5=TreeNode(9)
    t6=TreeNode(11)
    t7=TreeNode(13)
    t1.left=t2
    t1.right=t3
    t2.left=t4
    t2.right=t5
    t3.left=t6
    t3.right=t7
    test=latOder(t1)
    print(test)
    print(achieve(test))
```

## 排序二叉树to排序双向链表

```python
'''
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表
要求：不能创建任何新的结点，只能调整树中，结点指针的指向。
思路：

'''
def SearchTree2Link(root):
    if root==None:
        return None

    leftNode=SearchTree2Link(root.left)
    rightNode=SearchTree2Link(root.right)
    backNode=leftNode

    if leftNode:
        leftNode=findleft(leftNode)
    else:
        backNode=root
    
    root.left=leftNode
    root.right=rightNode
    if leftNode!=None:
        leftNode.right=root
    if rightNode!=None:
        rightNode.left=root
    return backNode

def findleft(root):
    node=root
    while node.right:
        node=node.right
    return node
if __name__=="__main__":
    t1=TreeNode(10)
    t2=TreeNode(8)
    t3=TreeNode(12)
    t4=TreeNode(7)
    t5=TreeNode(9)
    t6=TreeNode(11)
    t7=TreeNode(13)
    t1.left=t2
    t1.right=t3
    t2.left=t4
    t2.right=t5
    t3.left=t6
    t3.right=t7
```

## 二叉搜索树第k小的结点

```python
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
```



## 序列化与反序列化

```python
'''
实现序列化和反序列二叉树
'''
def serialize(root):
    tmpNode=root
    stack=[]
    res=[]
    while tmpNode or stack:
        while tmpNode:
            res.append(str(tmpNode.val))
            stack.append(tmpNode)
            tmpNode=tmpNode.left
        res.append('#')
        tmpNode=stack.pop()
        tmpNode=tmpNode.right
    return ' '.join(res)
def deserialize(strs):
    strlist=strs.split(' ')
    def depreOrder():
        if strlist==[]:
            return None
        rootVal=strlist.pop(0)
        if rootVal=='#':
            return None
        node=TreeNode(int(rootVal))
        node.left=depreOrder()
        node.right=depreOrder()
        return node
    root=depreOrder()
    return root
def preOder(root):
    if root ==None:
        return None
    stack=[]
    tmpNode=root
    res=[]
    while tmpNode or stack:
        while tmpNode:
            res.append(tmpNode.val)
            stack.append(tmpNode)
            tmpNode=tmpNode.left
        node=stack.pop()
        #print(node.val)
        tmpNode=node.right
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
    res=serialize(t1)
    print('序列化：',res)
    res0=deserialize(res)
    res1=preOder(res0)
    print('反序列化：',res1)
```

## 指定路径和

```python
'''
输入一棵二叉树和一个整数，打印出二叉树中结点值得和为输入整数的所有路径。
路径定义为从数的根节点开始往下一直到叶节点所经过的结点形成的一条路径了。
（注意：在返回值得list中，数组长度大的数组靠前）
思路：
1.通过广度优先遍历，将每个路径的上的结点加入到newTmpArrList上，
2.然后把同层的所有路径加到queueArrList上，
3.到了叶子结点时，计算queueArrList的元素tmpArrList上结点的和是否等于目标值，如果等于则添加到数组的第一个，因为路径长的放前面。
'''
import copy
def achieve(root,summary):
    queuelevel=[root]
    queueArrList=[[root.val]]
    res=[]
    while queuelevel:
        tmpNode=queuelevel.pop(0)
        tmpArrList=queueArrList.pop(0)
        if tmpNode.left==None and tmpNode.right==None:
            s=0
            for i in tmpArrList:
                s+=i
            if s==summary:
                res.insert(0,tmpArrList)
        if tmpNode.left:
            queuelevel.append(tmpNode.left)
            newTmpArrList=copy.copy(tmpArrList)
            newTmpArrList.append(tmpNode.left.val)
            queueArrList.append(newTmpArrList)
        if tmpNode.right:
            queuelevel.append(tmpNode.right)
            newTmpArrList=copy.copy(tmpArrList)
            newTmpArrList.append(tmpNode.right.val)
            queueArrList.append(newTmpArrList)
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
    res=achieve(t1,8)
    for i in res:
        print(i)
```

## 中位数

```python
'''
数据流的中位数
中位数是有序列表中间的数。如果列表长度是偶数，中位数则是中间两个数的平均值。
设计一个支持以下两种操作的数据结构：
void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。
思路：
1.这种动态（流动）的数据，如果使用数组存储，那么每次新进来一个数据都进行排序的话，效率很低。
处理动态数据来说，一般使用的数据结构是栈、队列、二叉树、堆。在这里还是采用最大最小堆。
2.建立一个类，它有两个属性数组，一个最大堆（用于存储数据流的较小的数），一个最小堆（用于存储数据流较大的数），
3.轮流往最大堆和最小堆放数（保证两个对各占一半的数据，中位数必在堆顶形成），在放之前要与另一个堆比较，
    与最小堆比较的时候，查看要放的数是不是比根节点的值大，如果大，此数放入最小堆，而根结点的值放入最大堆。
    与最大堆比较的时候，查看要放的数是不是比根节点的值小，如果小，此数放入最大堆，而根结点的值放入最小堆。
4.如果输入奇数个数，中位数在最大堆的堆顶。如果输入偶数个数，中位数等于（最大堆堆顶+最小堆堆顶）/2
'''
class DataFlowMedium():
    def __init__(self):
        self.maxHeap=[]
        self.minHeap=[]
        self.count=0
    def addNum(self,num):
        maxHeap=self.maxHeap
        minHeap=self.minHeap
        count=self.count
        tmp=0
        if count%2==1:#往最小堆放数，要与最大堆的堆顶做比较
            if num<maxHeap[0]:
                tmp=maxHeap[0]
                self.adjustMaxHeap(num)
            else:
                tmp=num
            self.addMinHeap(tmp)
        else:#往最大堆添数，要与最小堆堆顶做比较
            if maxHeap==[]:
                tmp=num
            else:
                if minHeap==[]:
                    tmp=num
                else:
                    if minHeap[0]<num:
                        tmp=minHeap[0]
                        self.adjustMinHeap(num)
                    else:
                        tmp=num
            self.addMaxHeap(tmp)
        self.count=count+1
    def getMedium(self):
        if self.count%2==1:
            return self.maxHeap[0]
        else:
            res=(self.maxHeap[0]+self.minHeap[0])/2
            return res
    def addMaxHeap(self,num):#和findKmin中的create相同
        maxHeap=self.maxHeap
        maxHeap.append(num)
        curIndex=len(maxHeap)-1
        while curIndex>0:
            parentIndex=(curIndex-1)>>1
            if maxHeap[parentIndex]<maxHeap[curIndex]:
                maxHeap[curIndex],maxHeap[parentIndex]=maxHeap[parentIndex],maxHeap[curIndex]
                curIndex=parentIndex
            else:
                break
    def addMinHeap(self,num):
        minHeap=self.minHeap
        minHeap.append(num)
        curIndex=len(minHeap)-1
        while curIndex>0:
            parentIndex=(curIndex-1)>>1
            if minHeap[curIndex]<minHeap[parentIndex]:
                minHeap[curIndex],minHeap[parentIndex]=minHeap[parentIndex],minHeap[curIndex]
                curIndex=parentIndex
            else:
                break
    def adjustMaxHeap(self,num):
        maxHeap=self.maxHeap
        length=len(maxHeap)
        if num<maxHeap[0]:
            maxHeap[0]=num
            curIndex=0
            while curIndex<length:
                leftChildIndex=2*curIndex+1
                rightChildIndex=2*curIndex+2
                largerIndex=0
                if rightChildIndex<length:
                    if maxHeap[rightChildIndex]<maxHeap[leftChildIndex]:
                        largerIndex=leftChildIndex
                    else:
                        largerIndex=rightChildIndex
                elif leftChildIndex<length :
                    largerIndex=leftChildIndex
                else:
                    break
                maxHeap[curIndex],maxHeap[largerIndex]=maxHeap[largerIndex],maxHeap[curIndex]
                curIndex=largerIndex
    def adjustMinHeap(self,num):
        minHeap=self.minHeap
        length=len(minHeap)
        if minHeap[0]<num:
            minHeap[0]=num
            curIndex=0
            while curIndex<length:
                leftChildIndex=2*curIndex+1
                rightChildIndex=2*curIndex+2
                smallerIndex=0
                if rightChildIndex<length:
                    if minHeap[rightChildIndex]<minHeap[leftChildIndex]:
                        smallerIndex=rightChildIndex
                    else:
                        smallerIndex=leftChildIndex
                elif leftChildIndex<length:
                    smallerIndex=leftChildIndex
                else:
                    break
                minHeap[curIndex],minHeap[smallerIndex]=minHeap[smallerIndex],minHeap[curIndex]
                curIndex=smallerIndex

if __name__=='__main__':
    dfm=DataFlowMedium()
    for item in [4,10,9,2,15,7,20,19,16,32,45,36]:
        dfm.addNum(item)
        mid=dfm.getMedium()
        # print(dfm.minHeap,dfm.maxHeap)
        print('中位数：%s' % (mid))
```



