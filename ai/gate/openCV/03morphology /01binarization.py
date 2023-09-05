import cv2
import numpy as np

img = cv2.imread('../data/practice.png')
# 将源图转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 全局阈值
# threshold(img,thresh,maxval,type)，
# type：
# THRESH_BINARY 高于阈值设为maxval，低于阈值设为0 THRESH_BINARY_INV，与前面相反
# THRESH_TRUNC 高于阈值设为maxal，低于阈值保持源值
# THRESH_TOZERO 高于阈值保持不变，低于阈值设为0, THRESH_TOZERO_INV 高于阈值设为0，低于阈值保持源值
# ret, res = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# 自适应阈值
# adaptiveThreshold(gray,maxValue,adaptiveMethod,type,blockSize,C)
res =cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
# adaptiveMethod：计算阈值的方法，ADAPTIVE_THRESH_MEAN_C计算邻近区域的平均值，ADAPTIVE_THRESH_GAUSSIAN_C高斯窗口加权平均值
# type：THRESH_BINARY，THRESH_BINARY_INV
# blockSize：光线影响较小可以设大一些
# C：常量，从计算的值中减去一个常量

cv2.imshow('img', img)
cv2.imshow('gray',gray)
cv2.imshow('res', res)

cv2.waitKey(0)
cv2.destroyAllWindows()