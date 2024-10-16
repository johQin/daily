import cv2
cv2.namedWindow('myWindow', cv2.WINDOW_NORMAL)        # 第一个参数为window的标识符，用于唯一确定窗口，WINDOW_NORMAL可以随意放大缩小窗口
cv2.resizeWindow('myWindow', 640, 480)      # 调整窗口大小
cv2.imshow('myWindow', 0)       # 显示窗口

key = cv2.waitKey(0)      # 窗口一直显示，焦点在显示的窗口时，直到收到键盘事件，关闭键盘,返回值为键盘的输入值
if (key & 0xFF) == ord('q'):
    print(key)
cv2.destroyAllWindows()     # 关闭所有窗口