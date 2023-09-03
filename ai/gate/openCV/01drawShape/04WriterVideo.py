import cv2
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('video', 640, 480)
cap = cv2.VideoCapture('../data/transport.flv')

# 创建VideoWriter
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
vw = cv2.VideoWriter('../data/reWriteTransport.mp4', fourcc, 30, (1280, 720))
# fps，frameSize要按照摄像头的参数来调整（ffprobe）
# VideoWriter::VideoWriter(const String& filename, int apiPreference, int _fourcc, double fps, Size frameSize, bool isColor)


while cap.isOpened():
    status, frame = cap.read()
    if status:
        cv2.imshow('video', frame)
        # 写帧
        vw.write(frame)
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
vw.release()
cv2.destroyAllWindows()