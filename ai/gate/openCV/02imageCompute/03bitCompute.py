# 位运算
import cv2
import numpy as np

img = np.zeros((200,200),np.uint8)
img2 = np.zeros((200, 200), np.uint8)


# 非操作
# img[50:150, 50:150] = 255       # 中心白，四周黑
# not_img = cv2.bitwise_not(img)
# cv2.imshow('img', img)
# cv2.imshow('not_img', not_img)

# 与操作
# img[:] = 255
# img2[:] = 255
# img[0:100, 0:100] = 0
# img2[100:200, 100:200] = 0
# and_img = np.bitwise_and(img,img2)
# cv2.imshow('img', img)
# cv2.imshow('img2', img2)
# cv2.imshow('and_img', and_img)

# 或操作
# img[0:100, 0:100] = 255
# img2[100:200, 100:200] = 255
# or_img = np.bitwise_or(img,img2)
# cv2.imshow('img', img)
# cv2.imshow('img2', img2)
# cv2.imshow('or_img', or_img)

# 异或，值不同为1,否则为0
img[0:100, 0:100] = 255
img2[100:200, 100:200] = 255
xor_img = np.bitwise_xor(img,img2)
cv2.imshow('img', img)
cv2.imshow('img2', img2)
cv2.imshow('xor_img', xor_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
