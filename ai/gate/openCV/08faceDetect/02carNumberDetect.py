import cv2
import numpy as np
import pytesseract

# 创建Haar级联器
# 车牌
carer = cv2.CascadeClassifier('../data/haarcascades/haarcascade_russian_plate_number.xml')

img = cv2.imread('../data/car_number.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cars = carer.detectMultiScale(gray, 1.1, 5)
for (x,y,w,h) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+h), (255, 0, 0), 2)
    roi = gray[y:y+h, x:x+w]
    ret, roi_bin = cv2.threshold(roi,0,255,cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    car_num = pytesseract.image_to_string(roi,lang="chi_sim+eng",config='--psm 8 --oem 3')
    print("车牌号：", car_num)


cv2.imshow('detect',img)
cv2.waitKey(0)
cv2.destroyAllWindows()