# 图像低通滤波
import cv2
import numpy as np

img = cv2.imread('../data/mother.jpg')
kernel = np.ones((5, 5), np.float32) / 25     # 对每个5x5的区域，做一个平均计算，平滑处理，模糊化处理
# res = cv2.filter2D(img, -1, kernel)

# 方盒滤波：boxFilter(src, ddepth, ksize, dst, anchor, normalize, borderType)
# 等于 np.ones((n, m), np.float32) * a。当normalize = true，a = 1/(n * m), 当为false时，a = 1

# 模糊滤波：cv2.blur(src, ksize, anchor, borderType)，
# 当方盒滤波的normalize为true时，方盒滤波核模糊滤波等价
# res = cv2.blur(img, (5,5))

# 高斯滤波
# GaussianBlur(src,kernel,sigmaX,sigmaY)，sigma的偏差，高斯分布公式中的sigma
# kernel 的所有权重之和为1，核中心的权重最大。
# res = cv2.GaussianBlur(img, (3, 3), sigmaX=1)

# 中值滤波
# medianBlur(src, ksize)
# res = cv2.medianBlur(img, 5)

# 双边滤波
# cv2.bilateralFilter(src,d,sigmaColor,sigmaSpace)  sigmaColor像素值域方差, sigmaSpace：空间域方差
res = cv2.bilateralFilter(img, 7, 20, 50)

cv2.imshow('img', img)
cv2.imshow('res', res)


cv2.waitKey(0)
cv2.destroyAllWindows()