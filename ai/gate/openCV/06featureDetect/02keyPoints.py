import cv2
import numpy as np

img = cv2.imread('../data/shudu.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 因为版权问题，opencv-contrib-python 3.4以上的版本都不支持SIFT和SURF了，解决方案是切换到3.4的版本

# SIFT对象
# sift = cv2.xfeatures2d.SIFT_create()
# 探测关键点
# kp = sift.detect(gray,None)
# 关键点中包含的信息：位置，大小和方向

# 计算关键点和其描述子
# kp,des=sift.compute(img,kp)
# opencv提供了更好的api
# sift.dectdetectAndCompute(src,mask)
# kp, desc = sift.detectAndCompute(gray,None)
# 其作用是进行特点匹配
# print(desc[0])


# # SURF对象
# surf = cv2.xfeatures2d.SURF_create()
# # 特征检测和特征描述子
# kp, desc = surf.detectAndCompute(gray,None)


# ORB
orb = cv2.ORB_create()
kp, desc = orb.detectAndCompute(gray,None)



# 绘制keypoints
cv2.drawKeypoints(gray, kp, img)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()