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