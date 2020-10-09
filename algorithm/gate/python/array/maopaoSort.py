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