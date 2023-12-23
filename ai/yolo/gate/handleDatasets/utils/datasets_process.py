
import cv2
import os
import matplotlib.pyplot as plt
from .colorsPanel import colorsPanelList



# 数据预处理，将WiderPerson.zip中的数据转为yolo可以训练的格式
#  unzip WiderPerson.zip -d WiderPerson

# 从官方的read.md文件可以了解到，标记目标类别数量共5类：pedestrians（行人），riders（骑行的人），partially-visible persons（存在遮挡的人），ignore regions（忽略的区域），crowd（人群）
# ########## Annotation Format ##########
# Each image of training and valiadation subsets in the "./Images" folder (e.g., 000001.jpg) has a corresponding annotation text file in the "./Annotations" folder (e.g., 000001.jpg.txt). The annotation file structure is in the following format:
#     '''
#     < number of annotations in this image = N >
#     < anno 1 >
#     < anno 2 >
#     ......
#     < anno N >
#     '''
# where one object instance per row is [class_label, x1, y1, x2, y2], and the class label definition is:
#     '''
#     class_label =1: pedestrians
#     class_label =2: riders
#     class_label =3: partially-visible persons
#     class_label =4: ignore regions
#     class_label =5: crowd
#     '''

# 读取标注文件，转为yolo格式
def convert2yolo(imgId, datasetsRoot="./datasets", srcDatasets = "", savePathPrefix='./person_data/labels/train/',toOne=True, labelIndexOffset = 0 ):
    '''
    @param imgId: 图片id
    @param savePathPrefix: 保存的yolo格式标注文件路径前缀
    @return yoloLabelFile: yolo格式的标注文件路径
    '''
    # 打开标注文件
    lines = []
    oriLabelFile = datasetsRoot + f'/./{srcDatasets}/Annotations/' + imgId + '.jpg.txt'
    with open(oriLabelFile, 'r') as f:
        lines = f.readlines()
        # 转为list
        lines = [line.strip() for line in lines]
    # 读取标注
    boxes = lines[1:] # [class_label, x1, y1, x2, y2]
    boxes = [box.split(' ') for box in boxes]

    # 读取标注文件对应的图片
    imgFile = datasetsRoot + f'/./{srcDatasets}/Images/' + imgId + '.jpg'
    img = cv2.imread(imgFile)
    # 转为yolo格式：类别id、x_center y_center width height，归一化到0-1，保留6位小数
    yolo_boxes = []
    img_h, img_w, _ = img.shape
    for box in boxes:
        class_label = int(box[0]) - labelIndexOffset # 从0开始，0:pedestrians, 1:riders, 2:partially-visible persons, 3:ignore regions, 4:crowd
        x1, y1, x2, y2 = [int(i) for i in box[1:]]
        if toOne:
            x_center = round((x1 + x2) / 2 / img_w, 6)
            y_center = round((y1 + y2) / 2 / img_h, 6)
            width = round((x2 - x1) / img_w, 6)
            height = round((y2 - y1) / img_h, 6)
        else:
            x_center = round((x1 + x2) / 2 , 6)
            y_center = round((y1 + y2) / 2 , 6)
            width = round((x2 - x1) , 6)
            height = round((y2 - y1) , 6)
        yolo_boxes.append([class_label, x_center, y_center, width, height])

    # 写入txt文件
    # 生成yolo格式的标注文件，类似：./person_data/labels/train/000001.txt
    yoloLabelFile = datasetsRoot + "/" +  savePathPrefix + imgId + '.txt'
    with open(yoloLabelFile, 'w') as f:
        for yolo_box in yolo_boxes:
            f.write(' '.join([str(i) for i in yolo_box]) + '\n')
    if os.path.exists(yoloLabelFile):
        return yoloLabelFile
    else:
        return None


# 根据yolo格式的标注文件，在图片上绘制
def yoloDraw(img, yoloLabelFile, classLens = 85):
    '''
    @param img: 图片
    @param yoloLabelFile: yolo格式的标注文件路径
    '''
    img_copy = img.copy()
    # 生成5类标签对应的颜色
    color_dict = {}
    for i in range(classLens):
        color_dict[i] = colorsPanelList[i]
    with open(yoloLabelFile, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        boxes = [line.split(' ') for line in lines]
        for box in boxes:
            class_label = int(box[0])
            x_center, y_center, width, height = [float(i) for i in box[1:]]
            x1 = int((x_center - width / 2) * img_copy.shape[1])
            y1 = int((y_center - height / 2) * img_copy.shape[0])
            x2 = int((x_center + width / 2) * img_copy.shape[1])
            y2 = int((y_center + height / 2) * img_copy.shape[0])
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color_dict[class_label], 2)
            cv2.putText(img_copy, str(class_label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,1, color_dict[class_label], 2)
    plt.imshow(img_copy[:,:,::-1])
    plt.axis('off')
    plt.show()