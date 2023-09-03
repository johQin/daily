import cv2
img = cv2.imread('../data/bus.jpg')
print("(分辨率x，分辨率y，通道数):",img.shape)    # (427, 640, 3)
print("图像大小：分辨率x * 分辨率y * 通道数 = {}".format(img.size))   # 819840
print('图像的位深：{}'.format(img.dtype))     # uint8