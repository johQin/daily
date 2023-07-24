# yolov5

yolov5（you only look once，version 5）是基于python环境，在pytorch机器学习框架上，一个开源的**目标检测**模型系列。

[yolo 结合 deepsort 实现目标跟踪](https://blog.csdn.net/Albert_yeager/article/details/129321339)

[pytorch gpu 安装](https://zhuanlan.zhihu.com/p/612181449)

# [0 初识](https://zhuanlan.zhihu.com/p/558477653)

## 0.1 项目结构

![](./legend/yolov5项目结构.png)

├── data：主要是存放一些超参数的配置文件（这些文件（yaml文件）是用来配置训练集和测试集还有验证集的路径的，其中还包括目标检测的种类数和种类的名称）；还有一些官方提供测试的图片。如果是训练自己的数据集的话，那么就需要修改其中的yaml文件。但是自己的数据集不建议放在这个路径下面，而是建议把数据集放到yolov5项目的同级目录下面。

![](./legend/data结构.png)

├── models：里面主要是一些网络构建的配置文件和函数，其中包含了该项目的四个不同的版本，分别为是s、m、l、x。从名字就可以看出，这几个版本的大小。他们的检测测度分别都是从快到慢，但是精确度分别是从低到高。这就是所谓的鱼和熊掌不可兼得。如果训练自己的数据集的话，就需要修改这里面相对应的yaml文件来训练自己模型。

![](./legend/models结构.png)

├── utils：存放的是工具类的函数，里面有loss函数，metrics函数，plots函数等等。

![](./legend/utils结构.png)

├── weights：放置训练好的权重参数。

- 里面存放了一个download_weights.sh，可以通过sh去下载权重。

- 也可以手动去下载，权重下载地址：https://github.com/ultralytics/yolov5/releases/tag/v7.0

- ```bash
  https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
  ```

├── detect.py：利用训练好的权重参数进行目标检测，可以进行图像、视频和摄像头的检测。

├── train.py：训练自己的数据集的函数。

├── test.py：测试训练的结果的函数。

├──requirements.txt：这是一个文本文件，里面写着使用yolov5项目的环境依赖包的一些版本，可以利用该文本导入相应版本的包。

## 0.2 coco数据集

```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
```

