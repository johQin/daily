import cv2
import numpy as np

img = np.zeros((480,640,3), np.uint8)
# 线
cv2.line(img, (10, 50), (50, 100), (0, 0, 255), 2, -1)    # 起始点，结束点，color，粗细，线型

# 多边形
pts = np.array([(300,10), (150,100), (450, 100)], np.int32)     # 必须是32位的
cv2.polylines(img, [pts], True, (0, 0, 255))    # pts, 多边形是否闭合，color
# 多边形填充
cv2.fillPoly(img, [pts], (255,0,0))

# 长方形
cv2.rectangle(img, (100,0), (120,25), (0,0,255), -1)

# 圆
cv2.circle(img, (300, 300), 100, (0, 0, 255))               # center, r，color

# 椭圆
cv2.ellipse(img,(300, 300), (100,50), 0, 90, 180, (0,0,255), -1)  # center,(长轴/2，短轴/2），长轴startangle，startangle，endangle, color，线型填充

# 文本
cv2.putText(img, "hello world", (10, 400), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0)) # text,start_point,font,font_size,color。中文会出现点问题
cv2.imshow('draw', img)
cv2.waitKey(0)
cv2.destroyAllWindows()