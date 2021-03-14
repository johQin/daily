import numpy as np
arr1 = np.arange(8).reshape(2,2,2)
arr2 = np.arange(8,16).reshape(2,2,2)
print(arr1)
print(arr2)
print(np.concatenate(arr1,arr2))