import cv2
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('video', 640, 480)

# 获取捕获对象

# 获取摄像头
# cap = cv2.VideoCapture(0)
# cv2.VideoCapture(camera_id)
# 其默认值为-1，表示随机选取一个摄像头；如果有多个摄像头，则用数字“0”表示第1个摄像头，用数字“1”表示第2个摄像头，以此类推。

# 获取视频文件
cap = cv2.VideoCapture('../data/transport.flv')
# cv2.VideoCapture("file_name")
# 读取视频文件

while cap.isOpened():       # isOpened可以判定摄像头或文件是否打开
    # 读取视频帧
    status, frame = cap.read()  # status在读到帧时为true

    if status:
        # 显示视频帧，如果视频帧的分辨率大于window的窗口大小，则会被撑大
        cv2.imshow('video', frame)
    else:
        break

    key = cv2.waitKey(10)    # 等待 n ms，这里可以通过ffprobe 去探测视频的帧数，以调整waitkey的时间，让视频不过快或过慢
    if key & 0xFF == ord('q'):
        break



# 释放摄像头或视频文件资源
cap.release()
cv2.destroyAllWindows()