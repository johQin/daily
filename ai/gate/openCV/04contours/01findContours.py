import cv2
import numpy as np

# img = cv2.imread('../data/contours.png')
img = cv2.imread('../data/多边形逼近和凸包.jpg')
fil = cv2.bilateralFilter(img, 7, 20, 50)
# 将源图转为灰度图
gray = cv2.cvtColor(fil, cv2.COLOR_BGR2GRAY)
# 二值化
ret, binImg = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
# 查找轮廓
contours, hierarchy = cv2.findContours(binImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓
res = cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
# contourIdx, -1 表示绘制最外层的轮廓
# thickness，线宽，-1表示填充

# 轮廓面积
area = cv2.contourArea(contours[0])
print(area)

# 轮廓周长
lens = cv2.arcLength(contours[0], True)    # close是否是闭合的轮廓
print(lens)

def drawShape(src, points):
    i = 0
    while i < len(points):
        x, y = points[i]
        if i + 1 == len(points):
            x1,y1 = points[0]
        else:
            x1,y1 = points[i+1]
        cv2.line(src,(x,y),(x1,y1), (0,255,0), 2)
        i = i + 1

# 多边形逼近
# approxPolyDP(curve,epsilon,closed)，curve轮廓，closed是否需要闭合
# epsilon 描点精度
approx = cv2.approxPolyDP(contours[0], epsilon=20, closed=True)
approxRSP = approx.reshape(approx.shape[0],approx.shape[2])
drawShape(img, approxRSP)

# 凸包
# convexHull(points,clockwise)
hull = cv2.convexHull(contours[0])
hullRSP = hull.reshape(hull.shape[0], hull.shape[2])
drawShape(img, hullRSP)



cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()