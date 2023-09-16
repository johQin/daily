import cv2
import numpy as np

cap = cv2.VideoCapture()

# MOG
# cv2.bgsegm.createBackgroundSubtractorMOG(history, nmixtures, backgroundRatio, noiseSigma)
# history：默认200，在进行建模的时候需要参考多少毫秒
# nmixtures 高斯范围值，默认5，将一阵图像分为5x5的小块
# backgroundRatio：背景比率，默认0.7，背景占整个图像的比例
# noiseSigma：默认0,自动降噪
# mog = cv2.bgsegm.createBackgroundSubtractorMOG()

# MOG2
# cv2.createBackgroundSubtractorMOG2(history,detectShadows)
# detectShadows 是否检测阴影，默认为True
# 好处可以计算出阴影
# 缺点可以产生横多噪点
mog = cv2.createBackgroundSubtractorMOG2()

# GMG
# cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames, decisionThreshold)
# initializationFrames： 初始参考帧数，默认120
# 好处，可以算出阴影部分，同时减少噪点
# 缺点，initializationFrames如果采用默认值，会有延迟，但可以改小一点
# mog = cv2.createBackgroundSubtractorGMG
while True:
    ret, frame = cap.read()
    fgmask = mog.apply(frame)
    cv2.imshow('img', fgmask)
    k = cv2.waitKey(10)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()