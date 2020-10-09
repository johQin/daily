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
