import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../data/coin.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# THRESH_OTSU 自适应阀值，这样硬币圆中间就干净多了
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cv2.imshow('threh', thresh)

# 开运算
kernel = np.ones((3,3),np.int8)
open1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 获取背景，膨胀
bg = cv2.dilate(open1, kernel, iterations=1)

# 获取前景物体
# 计算非0值距离它最近0值的距离
# distanceType = DIST_L1（|x1-x2| + |y1-y2|）,DIST_L2(平方差开根号)
# maskSize： 一般来说，L1对应用3（3x3），L1对应用5（5x5）
# cv2.distanceTransform(bg, distanceType=, maskSize=)
dist = cv2.distanceTransform(open1, distanceType=cv2.DIST_L2, maskSize=5)
# plt.imshow(dist, cmap='gray')
# plt.show()
ret, fg = cv2.threshold(dist, 0.7 * dist.max(), 255, cv2.THRESH_BINARY)

# 获取未知区域
fg = np.uint8(fg)
unknow = cv2.subtract(bg, fg)

# 创建连通域
# 求连通域，求非0元素的所有连通域，将具有相同像素值且相邻的像素找出来并标记。https://zhuanlan.zhihu.com/p/145449066
# connectivity：相邻的哪几个方向，4（上下左右），8（上下左右斜对角）
# cv2.connectedComponents(img,connectivity=)
ret, marker = cv2.connectedComponents(fg)
marker = marker + 1
marker[unknow == 255] = 0

# 进行图像分割
result = cv2.watershed(img,marker)
img[result == -1] = [255, 255, 0]

cv2.imshow('fg', fg)
cv2.imshow('bg', bg)
cv2.imshow('unknow', unknow)
cv2.imshow('result', img)

cv2.waitKey(0)
cv2.destroyAllWindows()