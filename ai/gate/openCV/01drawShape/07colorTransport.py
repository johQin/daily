import cv2
cv2.namedWindow('colorConversion',cv2.WINDOW_NORMAL)
girl = cv2.imread('../data/girl.jpg')
def onColorChange(ha):
    pass

# opencv-4.8.0/modules/imgproc/include/opencv2/imgproc.hpp
# enum ColorConversionCodes
colorSpace = [cv2.COLOR_BGR2BGRA, cv2.COLOR_BGR2BGRA, cv2.COLOR_BGR2GRAY, cv2.COLOR_BGR2HSV_FULL, cv2.COLOR_BGR2YUV]
cv2.createTrackbar('curCorlor', 'colorConversion', 0, len(colorSpace)-1, onColorChange)

while True:
    index = cv2.getTrackbarPos('curCorlor', 'colorConversion')

    # 颜色空间转换
    girlTo = cv2.cvtColor(girl, colorSpace[index])

    cv2.imshow('colorConversion', girlTo)
    key = cv2.waitKey(100)
    if key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

