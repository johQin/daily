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