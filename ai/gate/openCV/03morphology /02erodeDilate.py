# 腐蚀和膨胀
import cv2
import numpy as np

img = cv2.imread('../data/word_white_bg_black.jpeg')        # 黑底白字
# img = cv2.imread('../data/word_black_bg_white.jpeg')        # 白底黑字

# 将源图转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.ones((3,3),np.uint8)
# kernel = np.zeros((7,7),np.uint8)     # 全0的核处理前后不变

# 获取卷积核
# cv2.getStructuringElement(type,size)
# type：
# MORPH_RECT 全一
# MORPH_ELLIPSE 椭圆形内是1,椭圆形外是0
# MORPH_CROSS 十字架上是1，十字架外是0
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7,7))
print(kernel)

# 腐蚀
# res = cv2.erode(gray,kernel, iterations=1)       # iterations 腐蚀的次数
# 膨胀
res = cv2.dilate(gray, kernel, iterations=2)


cv2.imshow('img', img)
cv2.imshow('gray',gray)
cv2.imshow('res', res)

cv2.waitKey(0)
cv2.destroyAllWindows()