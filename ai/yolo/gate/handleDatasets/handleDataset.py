import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import shutil
import tqdm
from pathlib import Path
from utils import *


def datasetsTest(srcDatasets = "WiderPerson"):
    test_imgId = '000395'
    datasets_root = Path("./datasets")
    # 读取并显示原始图
    imgFile = datasets_root / (srcDatasets + '/Images/' + test_imgId + '.jpg')
    # 显示
    img = cv2.imread(str(imgFile))
    plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    # 转为yolo格式
    yoloLabelFile = convert2yolo(test_imgId,datasetsRoot="./datasets", srcDatasets=f"{srcDatasets}", savePathPrefix=str(datasets_root),toOne=True,labelIndexOffset=1)
    classNames = ["pedestrians", "riders", "partially-visible persons", "ignore regions", "crowd"]
    # 生成classes.txt，每行写入一个类别，0:pedestrians, 1:riders, 2:partially-visible persons, 3:ignore regions, 4:crowd
    # 格式：0 pedestrians
    with open(datasets_root / 'classes.txt', 'w') as f:
        ctx = []
        for ind,cls in enumerate(classNames):
            ctx.append(f'{ind} {cls}\n')
        f.writelines(ctx)

    # 测试显示，也可以在LabelImg中查看
    yoloDraw(img, yoloLabelFile, classLens=5)

def datasetsHandle():
    datasets_root = Path("./datasets")
    srcDatasets = "WiderPerson"
    # 创建多级文件路径：./person_data/images/train, ./person_data/images/val, ./person_data/labels/train, ./person_data/labels/val
    dstDatasets = "person_data"
    personSet = datasets_root / "person_data"
    if personSet.exists():
        raise Exception(f"{str(personSet)} existed，请重新指定数据预处理根目录")
    imagesTrain = personSet / "images" / "train"
    imagesVal = personSet / "images" / "val"
    labelsTrain = personSet / "labels" / "train"
    labelsVal = personSet / "labels" / "val"

    personSet.mkdir(parents=True, exist_ok=True)
    imagesVal.mkdir(parents=True, exist_ok=True)
    imagesTrain.mkdir(parents=True, exist_ok=True)
    labelsVal.mkdir(parents=True, exist_ok=True)
    labelsTrain.mkdir(parents=True, exist_ok=True)

    '''
    yolo要求的格式之一
    ./person_data
        ├── images
        │   ├── train
        │   └── val
        └── labels
            ├── train
            └── val
    '''


    # 获取所有训练图片的文件名
    train_img_file_names = []
    with open(datasets_root / srcDatasets / 'train.txt', 'r') as f:
        train_img_file_names = f.readlines()
        train_img_file_names = [x.strip() for x in train_img_file_names]
    # 获取所有验证图片的文件名
    val_img_file_names = []
    with open(datasets_root / srcDatasets / 'val.txt', 'r') as f:
        val_img_file_names = f.readlines()
        val_img_file_names = [x.strip() for x in val_img_file_names]

    # 打印文件数量
    print('train_img_file_names:', len(train_img_file_names))
    print('val_img_file_names:', len(val_img_file_names))



    # 处理训练集
    for img_file_name in tqdm.tqdm(train_img_file_names, desc='train'):
        # 转为yolo格式
        yoloLabelFile = convert2yolo(img_file_name, datasetsRoot=str(datasets_root), srcDatasets=f"{srcDatasets}",
                     savePathPrefix=f'./{dstDatasets}/labels/train/', toOne=True, labelIndexOffset=1)
        if yoloLabelFile:
            # 复制图片到指定路径
            imgSrcFile = f'{str(datasets_root)}/{srcDatasets}/Images/' + img_file_name + '.jpg'
            imgDstFile = f'{str(datasets_root)}/{dstDatasets}/images/train/' + img_file_name + '.jpg'
            shutil.copy(imgSrcFile, imgDstFile)

    # 处理验证集
    for img_file_name in tqdm.tqdm(val_img_file_names, desc='val'):
        # 转为yolo格式
        yoloLabelFile = convert2yolo(img_file_name, datasetsRoot=str(datasets_root), srcDatasets=f"{srcDatasets}",
                                     savePathPrefix=f'./{dstDatasets}/labels/val/', toOne=True, labelIndexOffset=1)
        if yoloLabelFile:
            # 复制图片到指定路径
            imgSrcFile = f'{str(datasets_root)}/{srcDatasets}/Images/' + img_file_name + '.jpg'
            imgDstFile = f'{str(datasets_root)}/{dstDatasets}/images/val/' + img_file_name + '.jpg'
            shutil.copy(imgSrcFile, imgDstFile)

    # 检查文件数量
    print('train_img_file_names:', len(os.listdir('./person_data/images/train')))
    print('val_img_file_names:', len(os.listdir('./person_data/images/val')))
    print('train_label_file_names:', len(os.listdir('./person_data/labels/train')))
    print('val_label_file_names:', len(os.listdir('./person_data/labels/val')))

if __name__ == '__main__':
    datasetsHandle()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
