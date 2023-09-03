import cv2
import numpy as np

# 图像拆分通道
img1 = cv2.imread("../data/bus.jpg", flags=1)  # flags=1 读取彩色图像(BGR)
cv2.imshow("BGR", img1)  # BGR 图像

# BGR 通道拆分
bImg, gImg, rImg = cv2.split(img1)  # 拆分为 BGR 独立通道，这里三个通道通过imshow显示后，并不是三原色
cv2.imshow("rImg", rImg)  # 直接显示红色分量 rImg 显示为灰度图像

# 将单通道扩展为三通道
imgZeros = np.zeros_like(img1)  # 创建与 img1 相同形状的黑色图像
imgZeros[:, :, 2] = rImg  # 在黑色图像模板添加红色分量 rImg
cv2.imshow("channel R", imgZeros)  # 扩展为 BGR 通道

print(img1.shape, rImg.shape, imgZeros.shape)       # (427, 640, 3) (427, 640) (427, 640, 3)

# 合并通道
img2 = cv2.merge((bImg, gImg, rImg))
cv2.imshow('img2', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()  # 释放所有窗口
