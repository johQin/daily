import cv2
import numpy as np

# cv2.inpaint(src,mask, inpaintRadius, flags)
# mask 可以通过用人工绘制需要修复的位置
# inpaintRadius, 圆形领域的半径
# flags，INPAINT_NS，INPAINT_TELEA

img = cv2.imread('inpaint.png')
mask = cv2.imread('inpaint_mask.png',0)     # 加0, 变为灰度图
dst = cv2.inpaint(img,mask,5,cv2.INPAINT_TELEA)
cv2.imshow('src', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

