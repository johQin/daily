import numpy as np
arr1 = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]],
])
# 元组也可以生成ndarray
# arr2 = np.array((1, 2, 3))
# print(arr2)
print(arr1.ndim)
print(arr1.size)
print(arr1.itemsize)
print(arr1.shape)
print(arr1.dtype)
print(arr1.strides)
print(arr1.flags)
print(arr1.real)
print(arr1.imag)