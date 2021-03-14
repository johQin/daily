import numpy as np
arr1 = np.array([1, 2, 3])
arr2 = np.arange(12).reshape(4, 3)
print(arr2)
print(arr1 + arr2)

#以上广播机制的运用等同于
arr11 = np.tile(arr1, (4, 1))# 纵向平铺4次
print(arr11)
print(arr11+arr2)
