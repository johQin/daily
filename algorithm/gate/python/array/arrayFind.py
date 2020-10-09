
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
