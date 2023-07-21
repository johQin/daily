import numpy as np
x = np.array([3, 1, 2])
y = np.argsort(x) #[1 2 0]
print ('以排序后的顺序重构原数组：')
print (x[y])# [1 2 3]