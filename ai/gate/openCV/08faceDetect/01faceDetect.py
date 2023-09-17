import cv2
import numpy as np

# 创建Haar级联器
# 人正脸
facer = cv2.CascadeClassifier('../data/haarcascades/haarcascade_frontalface_default.xml')
# 人眼
eyer = cv2.CascadeClassifier('../data/haarcascades/haarcascade_eye.xml')

img = cv2.imread('../data/mother.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detectMultiScale(image, scaleFactor, minNeighbors)
# scaleFactor：缩放因子
# minNeighbors：最小的邻近像素
faces = facer.detectMultiScale(gray, 1.1, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0), 2)

    # 在人脸的基础上进行人眼识别
    roi_img = img[y:y+h,x:x+w]
    eyes = eyer.detectMultiScale(roi_img, 1.1, 5)
    for (xe, ye, we, he) in eyes:
        cv2.rectangle(roi_img, (xe, ye), (xe + we, ye + he), (0, 255, 0), 2)

cv2.imshow('detect',img)
cv2.waitKey(0)
cv2.destroyAllWindows()