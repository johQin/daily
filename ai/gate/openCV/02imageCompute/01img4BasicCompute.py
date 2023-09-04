# 图像的四则运算
import cv2
import numpy as np

# 只有shape一样的时候才能做四则运算
orgin = cv2.imread('../data/mother.jpg')
cv2.imshow('origin',orgin)

offset = np.ones(orgin.shape, np.uint8) * 100

# 加
res = cv2.add(offset,orgin)     # 图像变亮
cv2.imshow('add result', res)

# 减
subRes = cv2.subtract(orgin, offset)    # 图像变暗
cv2.imshow('substract result', subRes)

# 乘除
# cv2.multiply(a,b)
# cv2.divide(a,b)

cv2.waitKey(0)
cv2.destroyAllWindows()