import numpy as np
# 1.数组转置
# np.transpose(arr,axes)
# arr.T
arr1 = np.arange(12).reshape(3, 4)
print(np.transpose(arr1))
# [[ 0  4  8]
#  [ 1  5  9]
#  [ 2  6 10]
#  [ 3  7 11]]
print(arr1.T)
# [[ 0  4  8]
#  [ 1  5  9]
#  [ 2  6 10]
#  [ 3  7 11]]
print(arr1)

# 2.滚动数组
# np.rollaxis(arr,axis,start=0)
# 数组每个元素在多维空间的坐标按顺序每行一个元素，这样构成的矩阵，它的每列就是一个轴，第一列就是0号轴，第二列1号轴
# 将数组arr所对应的axis号轴 放在 start号轴的前面，start号轴后面的所有轴依次往后滚动一“列”
# 然后调整元素到对应坐标的位置上面去，形成的数组就是，rollaxis返回的数组
arr2 = np.arange(27).reshape(3, 3, 3)
print(arr2)
np.rollaxis(arr2, 2,1)

# 3.交换数组两个轴
# numpy.swapaxes(arr, axis1, axis2)
# 这里轴概念和上面的概念相同