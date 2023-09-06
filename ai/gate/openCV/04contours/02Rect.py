import cv2
import numpy as np

img = cv2.imread('../data/rect.png')
fil = cv2.bilateralFilter(img, 7, 20, 50)
# 将源图转为灰度图
gray = cv2.cvtColor(fil, cv2.COLOR_BGR2GRAY)
# 二值化
ret, binImg = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# 最小外接矩形
# RotatedRect = cv2.minAreaRect(points)
# RotatedRect：x,y,width,height,angle
mr = cv2.minAreaRect(contours[1])
box = cv2.boxPoints(mr)
box = np.int0(box)
cv2.drawContours(img,[box], 0, (0,0,255),2)


# 最大外接矩形
# Rect= cv2.boundingRect(array)
# Rect：x,y,width,height
x,y,w,h = cv2.boundingRect(contours[1])
cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)



cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()