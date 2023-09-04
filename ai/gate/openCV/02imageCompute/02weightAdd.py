# 图像溶合，两张图片按照权重重合在一起
# addWeighted(A,alpha,B,beta,gramma)
# alpha * A + beta * B + gramma
# alpha + beta = 1

import cv2
import numpy as np

# shape 要一致才能溶和
couple = cv2.imread('../data/couple.jpg')   # (640, 416, 3)
girl1 = cv2.imread('../data/girl1.png')     # (640, 416, 3)
print(couple.shape)
print(girl1.shape)

res = cv2.addWeighted(couple, 0.7, girl1, 0.3, 0)
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()