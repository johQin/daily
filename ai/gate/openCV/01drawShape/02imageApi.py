# opencv-4.8.0/modules/imgcodecs/include/opencv2/imgcodecs.hpp
# CV_EXPORTS_W Mat imread( const String& filename, int flags = IMREAD_COLOR );
import cv2
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
# 读取图片
girl = cv2.imread('../data/girl.jpg')
cv2.imshow('img', girl)
key = cv2.waitKey(0)
if key & 0xFF == ord('q'):
    print('hah')
else:
    cv2.imwrite("../data/123.png", girl)

cv2.destroyAllWindows()