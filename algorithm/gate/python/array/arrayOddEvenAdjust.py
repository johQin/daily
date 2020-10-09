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