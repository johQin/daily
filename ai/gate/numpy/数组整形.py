import numpy as np
arr1 = np.arange(9)
print(arr1)
print(arr1.reshape(3, 3))
print(arr1)
arr2 = arr1.reshape(3, 3)

for ele in arr2.flat:
    print(ele, end=',')
print('\n')
print(arr2)
print(arr2.flatten())
print(arr2)

arr3 = arr2.ravel()
print(arr3)
arr3[0] = 1
print(arr3)
print(arr2)