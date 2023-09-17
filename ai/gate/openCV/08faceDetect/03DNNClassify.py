import cv2
import numpy as np
from cv2 import dnn

# 导入模型，创建神经网络
# 模型下载：http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
# 配置参数：https://github.com/opencv/opencv_extra/blob/4.1.0/testdata/dnn/bvlc_googlenet.prototxt

config = "../data/dnn/bvlc_googlenet.prototxt"
model = "../data/dnn/bvlc_googlenet.caffemodel"
net = dnn.readNetFromCaffe(prototxt=config, caffeModel=model)

img = cv2.imread('../data/girl.jpg')
blob = dnn.blobFromImage(img,1.0, (224,224), (104,117,123))

net.setInput(blob)
r = net.forward()

# 读入目录
classes = []
with open('../data/dnn/synset_words.txt') as f:
    classes = [x [x.find(" ")+1:]for x in f]

order = sorted(r[0] , reverse=True)
z = list(range(3))
for i in range(0,3):
    z[i] = np.where(r[0] == order[i])[0][0]
    print("第{}项匹配：{} 类所在行：{} 可能性：{}".format(i + 1,classes[z[i]],z[i]+1,order[i]))