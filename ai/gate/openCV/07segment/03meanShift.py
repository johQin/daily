import cv2
import numpy as np

src = cv2.imread('../data/flower.png')
mean_img = cv2.pyrMeanShiftFiltering(src, sp=20, sr=30)

canny_filter = cv2.Canny(mean_img, 150, 300)
contours, _= cv2.findContours(canny_filter,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(src, contours, -1, (0,0,255), 2)
cv2.imshow('src',src)
cv2.imshow('res',mean_img)
cv2.imshow('canny_filter',canny_filter)
cv2.waitKey(0)
cv2.destroyAllWindows()