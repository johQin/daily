#setMouseCallbask(winname, callback, userdata)
#callback(event, x, y, flags, userdata)
#event 鼠标相关的事件
#flags 鼠标键和组合键
# opencv-4.8.0/modules/highgui/include/opencv2/highgui.hpp

import cv2
import numpy as np

cv2.namedWindow('mouse', cv2.WINDOW_NORMAL)
cv2.resizeWindow('mouse', 640, 480)

def mouseCallback(event,x,y,flags,userdata):
    print(event, x, y, flags, userdata)

cv2.setMouseCallback('mouse', mouseCallback, [1,2,3])

# 全0就是黑色
blackframe = np.zeros((360, 640, 3), np.uint8)     # 分辨率：360 * 640 ，rgb(3)

while True:
    cv2.imshow('mouse', blackframe)
    key = cv2.waitKey(100)
    if key & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
