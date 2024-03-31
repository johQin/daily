# 开，闭，形态学梯度，顶帽，黑帽
import cv2
import numpy as np

img = cv2.imread('../data/word_white_bg_black.jpeg')   # 黑底白字
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
# morphologyEx(src,op,kernel)，op：
# 开运算
# res = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)     # 噪点越大，核选择就越大
# 闭运算
# res = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
# 形态学梯度
# res = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
# 顶帽
# res = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
# 黑帽
res = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

cv2.imshow('img', img)
cv2.imshow('gray', gray)
cv2.imshow('res', res)

cv2.waitKey(0)
cv2.destroyAllWindows()