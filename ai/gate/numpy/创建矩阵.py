import numpy as np
arr1 = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]],
], dtype=np.float32)

arr2 = np.zeros((3, 4))
print(arr2)
arr3 = np.ones((3, 4))
print(arr3)

# np.arange(start, stop, step, dtype)
arr4 = np.arange(12)
print(arr4)
print(arr4.reshape((3, 4)))
print(arr4)# 原矩阵不发生变化

# np.linspace(start,stop,num)，在start与stop段内，分为num个点
arr5 = np.linspace(0, 10, 20)
print(arr5)