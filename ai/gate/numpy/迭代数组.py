import numpy as np
arr1 = np.arange(12).reshape(3, 4)
print(arr1)
for i in np.nditer(arr1):
    print(i, end=',')
print('\n')
for i in np.nditer(arr1, order="C"):
    print(i, end=',')
print('\n')
for i in np.nditer(arr1, order="F"):
    print(i, end=',')
print('\n')

# 默认情况下，nditer将视待迭代遍历的数组为只读对象（read-only），为了在遍历数组的同时，实现对数组元素值得修改，必须指定op_flags=['readwrite']模式：
for x in np.nditer(arr1, op_flags=['readwrite']):
    x[...] = 2 * x
print(arr1)

for x in np.nditer(arr1, flags=['external_loop'], op_flags=['readwrite']):
    print(x, end=',')
print('\n')
arr2 = np.arange(4)
arr3 = np.arange(12).reshape((3, 4))
for x, y in np.nditer([arr2, arr3]):
    print("%d:%d" % (x, y), end=',')