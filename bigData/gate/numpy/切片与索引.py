import numpy as np
print('arr1---------------')
arr1 = np.arange(10)
print(arr1)
print(arr1[1])
print(arr1[4:])
print(arr1[1:4])
print(arr1[1:8:2])
print('arr2---------------')
arr2 = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
print(arr2[...])
print(arr2[..., 1])
print(arr2[..., 1:])
print('arr3---------------')
arr3 = np.array([[1,  2],  [3,  4],  [5,  6]])
y = arr3[[0, 1, 2],  [0, 1, 0]]
print(y)
arr4 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
arr5 = arr4[
    [[0, 0], [3, 3]], # 行索引
    [[0, 2], [0, 2]] # 列索引
]
print(arr5)
print(4 % 2 == 1)
print(arr4 % 2)
print(arr4[arr4 % 2 ==1])
arr6 = np.arange(32).reshape((8, 4))
print (arr6[[4, 2, 0, 7]])