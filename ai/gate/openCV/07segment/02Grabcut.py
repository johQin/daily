import cv2
import numpy as np

class GrabCutApp:

    startX = 0
    startY = 0
    flag_rect = False # 鼠标左键是否按下
    rect = (0, 0, 0, 0)

    def onmouse(self,event,x,y,flags,params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.flag_rect = True
            self.startX = x
            self.startY = y
        elif event ==cv2.EVENT_LBUTTONUP:
            self.flag_rect = False
            cv2.rectangle(self.img, (self.startX,self.startY),(x,y),(0,0,255),3)
            self.rect = (min(self.startX,x), min(self.startY,y), abs(self.startX - x), abs(self.startY - y))
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.flag_rect:
                self.img = self.img2.copy()
                cv2.rectangle(self.img, (self.startX, self.startY), (x, y), (0, 255, 0), 3)
            pass

    def run(self):
        cv2.namedWindow('input')
        cv2.setMouseCallback('input', self.onmouse)

        self.img = cv2.imread('../data/lena.jpg')
        self.img2 = self.img.copy()     # 备份
        self.mask = np.zeros(self.img.shape[:2], np.uint8)
        self.output = np.zeros(self.img.shape, np.uint8)
        # 更新图的变化
        while True:
            cv2.imshow('input', self.img)
            cv2.imshow('output', self.output)
            key = cv2.waitKey(100)
            if key & 0xFF == ord('q'):
                break
            # 注意要在output窗口按g
            if key & 0xFF == ord('g'):
                bgdmodel = np.zeros((1,65), np.float64)
                fgdmodel = np.zeros((1,65), np.float64)
                # 获取分割的掩码
                cv2.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((self.mask == 1) | (self.mask == 3), 255, 0).astype('uint8')
            self.output = cv2.bitwise_and(self.img2, self.img2, mask=mask2)
        cv2.destroyAllWindows()

#
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount)
# mask：BGD——0背景，BGD——1前景，PR_BGD——2可能是背景，PR_FGD——3可能是前景
# Model: bgdModel，np.float64 1x65 (0,0,...)，fgdModel和bgd一样
# mode：GC_INIT_WITH_RECT从一个矩形框里扣，GC_INIT_WITH_MASK也可以从掩码里继续迭代
if __name__ == '__main__':
    ga = GrabCutApp()
    ga.run()
    print(12)