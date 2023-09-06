import cv2
import numpy as np

img = cv2.imread('../data/shudu.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Harris角点检测
# cornerHarris(src, blockSize, ksize, k, dst, borderType)
# blockSize检测窗口的大小
# ksize sobel的卷积核
# k权重系数，经验值，一般在0.02~0.04之间
# corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
# img[corners>0.01*corners.max()] = [0,0,255]

# shi-Tomasi角点检测
# goodFeaturesToTrack(src, maxCorners,qualityLevel,minDistance,mask)
# maxCorners 保留角点的最多个数
# qualityLevel 角点的质量，就是harris中的那个max，要大于这个质量，此能算作角点
# minDistance 角之间最小的欧氏距离，两个角小于这个距离，只保留一个角
# mask 需要检测的区域
# blockSize 检测窗口的大小
# useHarrisDetector 是否使用harris
# k默认是0.04
corners =cv2.goodFeaturesToTrack(gray, maxCorners = 1000,qualityLevel= 0.01,minDistance=10)
corners = np.intp(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3, (255,0 ,0), -1)
    

cv2.imshow('img', img)


cv2.waitKey(0)
cv2.destroyAllWindows()