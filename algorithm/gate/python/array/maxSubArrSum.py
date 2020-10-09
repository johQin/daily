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
