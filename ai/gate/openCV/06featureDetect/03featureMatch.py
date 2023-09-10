import cv2
import numpy as np

img1 = cv2.imread('../data/opencv_search.png')
img2 = cv2.imread('../data/opencv_orig.png')

# 灰度化
g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 创建sift特征检测器
sift = cv2.xfeatures2d.SIFT_create()
# 计算特征点和描述子
kp1, desc1 = sift.detectAndCompute(g1, None)
kp2, desc2 = sift.detectAndCompute(g2, None)

# # 暴力特征匹配器：创建匹配器
# bf = cv2.BFMatcher(cv2.NORM_L1)
# # 特征匹配
# match = bf.match(desc1, desc2)
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, match, None)


# flann特征匹配器：创建匹配器
index_params = dict(algorithm = 1, trees = 5)
search_params = dict(checks= 50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches =flann.knnMatch(desc1,desc2, k=2)
good = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        good.append(m)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,[good], None)


#
if len(good)>4:
    srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dstPts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    H,_ = cv2.findHomography(srcPts,dstPts)

cv2.imshow('imge3', img3)
cv2.waitKey(0)