# C++
# 浅拷贝
# Mat A;
# A = imread(file,IMREAD_COLOR);
# Mat B(A);
# 深拷贝
# cv::Mat::clone();
# cv::Mat::copyTo();

# 在python中深拷贝只暴露了一个copy
import cv2


img1 = cv2.imread('../data/bus.jpg')
# 浅拷贝， 直接赋值
img2 = img1
# 深拷贝
img3 = img1.copy()


img1[10:100, 10:100] = [0, 0, 255]
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()



