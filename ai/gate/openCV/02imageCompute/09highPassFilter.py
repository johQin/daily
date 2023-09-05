# 图像高通滤波
import cv2
import numpy as np

img = cv2.imread('../data/shudu.jpeg')

# 索贝尔滤波
# Sobel(src, ddepth, dx, dy, ksize)，ddepth位深，x方向求边缘，y方向求边缘
# # 索贝尔x求边缘
# resx = cv2.Sobel(img,cv2.CV_64F, 1, 0, ksize=5)
# # 索贝尔y求边缘
# resy = cv2.Sobel(img,cv2.CV_64F, 0, 1, ksize=5)
# # res = resx + resy
# res = cv2.add(resx, resy)

# 沙尔滤波
# # Scharr(src, ddepth, dx, dy)
# # 索贝尔x求边缘
# resx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
# # 索贝尔y求边缘
# resy = cv2.Scharr(img, cv2.CV_64F, 0, 1)
# # res = resx + resy
# res = cv2.add(resx, resy)

# 拉普拉斯滤波
# res = cv2.Laplacian(img, cv2.CV_64F, ksize=5)

# canny滤波，最牛
res = cv2.Canny(img,100,200)
cv2.imshow('img', img)
cv2.imshow('res', res)

cv2.waitKey(0)
cv2.destroyAllWindows()