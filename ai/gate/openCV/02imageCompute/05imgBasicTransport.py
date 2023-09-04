# 缩放，翻转，旋转

import  cv2
import numpy as np
couple = cv2.imread('../data/couple.jpg')
cv2.imshow('couple',couple)

# 缩放
# 缩放，resize(src,dst,dsize,fx,fy,interpolation)，
# dsize目标尺寸，fx方向x的缩放比例，fy方向y的缩放比例，指定了dsize就不会按照fx和fy的缩放比例执行，
# interpolation 插值算法：
# INTER_NEAREST：邻近插值，速度快，效果差
# INTER_LINEAR：双线性插值，源图中的四个点
# INTER_CUBIC：三次插值，源图中的16个点
# INTER_AREA：效果最好，最慢

# # small = cv2.resize(couple,(300,400),interpolation=cv2.INTER_NEAREST)
# small = cv2.resize(couple,None, fx= 0.7, fy = 0.7, interpolation=cv2.INTER_NEAREST)
# cv2.imshow('small couple', small)


# 图像的翻转
# flip(img,flipCode)
# flip: 0 上下，>0左右，<0上下左右

# reverse_vertical = cv2.flip(couple, -1)
# cv2.imshow('reverse_vertical', reverse_vertical)


# 图像的旋转
# rotate(img, rotateCode)
# ROTATE_90_CLOCKWISE，顺时针90
# ROTATE_180
# ROTATE_90_COUNTERCLOCKWISE，逆时针90

cv2.waitKey(0)
cv2.destroyAllWindows()
