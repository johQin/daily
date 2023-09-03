import  numpy as np
import cv2

cv2.namedWindow('numpy',cv2.WINDOW_NORMAL)

odim = np.array([2, 3, 4])
tdim = np.array([[4,5,6],[7,8,9]])

np.zeros((4,4,3),np.uint8)
np.ones((4,4,3),np.uint8)
np.full((4,4,3),255,np.uint8)
np.identity(4)          # 方形斜角单位阵
np.eye(5, 7, k = 3)   # 矩形斜角单位阵

# 访问和赋值
img = np.zeros((480,640,3),np.uint8)

# count = 10
# while count < 200:
#     img[count, 100] = 255     # 白色，等价于 img[count, 100] = [255,255,255]
#     img[count, 150, 2] = 255    # blue
#     img[count, 200, 1] = 255    # 绿色
#     count += 1


# 获取子矩阵，x_start:to_x, y_start:to_y
roi = img[100:300,100:200]      # 获取了源矩阵在开始点(100,100)，结束点(300,200)之间的矩阵
roi[:,:] = [0,0,255]        # 在源图上画了一个红色矩形
# 上面等价于 roi[:] = [0,0,255]

cv2.imshow('numpy', img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()


# 获取子矩阵
