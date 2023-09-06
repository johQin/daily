import cv2
import numpy as np

cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture('../data/vehicle.mp4')

min_w = 90
min_h = 90
# 检测线距离上方的距离
lineTop = 550   # 视频的分辨率是1280*720
# 检测线检测高度检测区间（高度偏移量）
lineOffset = 7

# 车辆统计数
carcount = 0

# 这个库在扩展包里，pip install opencv-contrib-python，记得先安装opencv，再安装这个扩展包，版本才能匹配
# 去背景，在时间维度上，多帧图像是静止的事物一般是静止的，有论文可查看
bgsubmog = cv2.bgsegm.createBackgroundSubtractorMOG()

kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def center(x,y,w,h):
    return x + int(w/2), y + int(h/2)
while cap.isOpened():

    status, frame = cap.read()

    if status:
        # 灰度
        cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # 去噪
        blur = cv2.GaussianBlur(frame, (3,3), 5)
        # 去背景
        mask = bgsubmog.apply(blur)
        # 腐蚀，去除背景中的白色噪点，比如说晃动的树和草
        erode = cv2.erode(mask, kernel)
        # 膨胀，去除目标上的黑色背景噪点
        dilate = cv2.dilate(erode, kernel, iterations=3)
        # 闭操作，去掉物体内部的小块，将整个目标连接成一个大目标（内部没有块）
        close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
        close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, h = cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # 画一条检测线
        cv2.line(frame, (100, lineTop), (1180, lineTop), (255, 255, 0), 2)
        # 存放当前帧有效车辆的数组
        cars = []
        for (i,c) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(c)

            # 通过宽高判断是否是有效的车辆
            isValidVehicle = w >= min_w and h >= min_h
            # 如果不是有效的车辆，直接跳过
            if not isValidVehicle:
                continue

            # 绘制车辆的方框
            cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,255),2)

            # 求车辆的几何中心点
            pc = center(x, y, w, h)

            cars.append(pc)
            #车辆的几何中心在检测线上下lineOffset的范围内将会被统计
            for (x,y) in cars:
                if y >= lineTop - lineOffset and y <= lineTop + lineOffset:
                    carcount += 1
                    print(carcount)

        cv2.putText(frame, "Cars Count:{}".format(carcount),(500,60),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
        cv2.imshow('video', frame)
    else:
        break

    key = cv2.waitKey(10)    # 等待 n ms，这里可以通过ffprobe 去探测视频的帧数，以调整waitkey的时间，让视频不过快或过慢
    if key & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()