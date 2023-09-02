import cv2
import numpy as np

cv2.namedWindow('trackbar',cv2.WINDOW_NORMAL)
cv2.resizeWindow('trackbar',640,480)

def trackbarCallback(ha):
    pass

# createTrackbar(trackbarname, winname,cur_value,max_count,callback,userdata)
# getTrackbarPos(trackbarname, winname)返回当前trackbar的值
cv2.createTrackbar('R', 'trackbar', 0, 255, trackbarCallback)
cv2.createTrackbar('G', 'trackbar', 0, 255, trackbarCallback)
cv2.createTrackbar('B', 'trackbar', 0, 255, trackbarCallback)

img = np.zeros((480,480,3), np.uint8)

while True:

    # 获取颜色
    r = cv2.getTrackbarPos('R', 'trackbar')
    g = cv2.getTrackbarPos('G', 'trackbar')
    b = cv2.getTrackbarPos('B', 'trackbar')
    img[:] = [b, g, r]

    cv2.imshow('trackbar', img)
    key = cv2.waitKey(100)

    if key & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
