# 透视变换——视角变换，坐标系变换
import cv2
import numpy as np

img = cv2.imread('../data/paper_perspective.png')
h,w,ch = img.shape

# A4纸在三维图像到二维平面的变换
src = np.float32([[154,151],[361,143],[149,266],[452,239]])
dst = np.float32([[150,150],[360,150],[150,250],[360,250]])

M = cv2.getPerspectiveTransform(src,dst)
res = cv2.warpPerspective(img,M,(w, h))

cv2.imshow('src',img)
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()