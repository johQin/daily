# 仿射变换Affine Transformation
# 仿射变换变化包括缩放（Scale、平移(transform)、旋转(rotate)、反射（reflection,对图形照镜子）、错切(shear mapping，感觉像是一个图形的倒影)
# 原来的直线仿射变换后还是直线，原来的平行线经过仿射变换之后还是平行线，这就是仿射
# warpAffine(src,M,dsize,flags,mode,value)
# M 变换矩阵
# dsize输出大小
# flags 插值算法
# mode 边界外推法标志
# value 填充边界的值

import cv2
import numpy as np

img = cv2.imread("../data/bus.jpg")
h,w,ch = img.shape

# 平移变换矩阵
# M = np.float32([[1,0,100],[0,1,-100]])
# res = cv2.warpAffine(img, M, (w, h))



# 获取转换矩阵 CV_EXPORTS_W Mat getRotationMatrix2D( Point2f center, double angle, double scale )
# center中心点(x,y)，angle旋转的角度（逆时针），scale缩放
# M = cv2.getRotationMatrix2D((w/2, h/2), 15, 0.3)

# 还可以通过图上的变化前的三个点，和变化后对应的三个点，来确定变换矩阵getAffineTransform(src,dst)
src = np.float32([[400,300],[500,300],[400,600]])
dst = np.float32([[200,400], [300,500], [100,700]])
M = cv2.getAffineTransform(src,dst)


# 如果想要修改新图像的尺寸需要修改这里的dsize
res = cv2.warpAffine(img, M, (w, h))

cv2.imshow('img', img)
cv2.imshow('trans', res)
cv2.waitKey(0)
cv2.destroyAllWindows()