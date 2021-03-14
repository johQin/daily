import numpy as np
arr1 = np.arange(9).reshape(3, 3)
print(arr1)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]
print(np.amin(arr1, 0))# [0 1 2]
print(np.amin(arr1, axis=0))# [0 1 2]
print(np.amin(arr1, 1))# [0 3 6]
print(np.ptp(arr1))
print(np.ptp(arr1, 0))#[6 6 6]
print(np.ptp(arr1, 1))#[2 2 2]

arr2 = np.array([[10, 7, 4], [3, 2, 1]])
print(arr2)

# 50% 的分位数，就是 arr2 里排序之后的中位数
print(np.percentile(arr2, 50))# 3.5
# axis 为 0，在纵列上求
print(np.percentile(arr2, 50, axis=0))# [6.5 4.5 2.5]
# axis 为 1，在横行上求
print(np.percentile(arr2, 50, axis=1))# [7. 2.]
# 保持维度不变
print(np.percentile(arr2, 50, axis=1, keepdims=True))
# [[7.]
#  [2.]]



