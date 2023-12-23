# yolov5

yolov5（you only look once，version 5）是基于python环境，在pytorch机器学习框架上，一个开源的**目标检测**模型系列。

[yolo 结合 deepsort 实现目标跟踪](https://blog.csdn.net/Albert_yeager/article/details/129321339)

[pytorch gpu 安装](https://zhuanlan.zhihu.com/p/612181449)

# [0 初识](https://zhuanlan.zhihu.com/p/558477653)

**yolov5 tagv5.0版本代码**

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

## 0.2 [GPU环境搭建](https://blog.csdn.net/qq_53357019/article/details/125725702)

### 0.2.1 安装nvidia显卡驱动、cuda toolkit、cudnn



**CUDA Toolkit** 是用于开发和运行基于 CUDA 的应用程序的软件包。它包含了编译器、库、工具和示例等组件，用于编写、构建和优化 CUDA 应用程序。CUDA Toolkit 还提供了与 GPU 相关的驱动程序和运行时库，以便在系统上正确配置和管理 GPU。这个库的主要目的是帮你封装好了很多的操作这个gpu ，也就是操作这个 cuda 驱动的库。

**cuDNN（CUDA Deep Neural Network library）**是 NVIDIA 为深度学习框架提供的加速库。它为深度神经网络的训练和推理提供了高性能的 GPU 加速支持。cuDNN 提供了一系列优化的算法和函数，用于加速卷积、池化、归一化等常用的深度学习操作。它与 CUDA 和 CUDA Toolkit 配合使用，提供了对深度学习框架（如TensorFlow、PyTorch等）的 GPU 加速能力。

[nvidia 显卡驱动 安装最顺的教程](https://zhuanlan.zhihu.com/p/302692454)，推荐查看

[选择显卡驱动版本和toolkit版本下载，不含安装报错的显卡驱动安装教程](https://blog.csdn.net/weixin_39928010/article/details/131142603)

[ubuntu cudnn 安装](https://blog.csdn.net/shanglianlm/article/details/130219640)

### 0.2.2 python 环境安装

[解决torch安装缓慢失败及其他安装包快速下载方法](https://blog.csdn.net/qq_35207086/article/details/123482458)

```bash
# 安装有些包的时候，很慢，可以通过清华源的方式修改
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.0.1

#（1）阿里云     https://mirrors.aliyun.com/pypi/simple/
#（2）豆瓣     https://pypi.douban.com/simple/
#（3）清华大学     https://pypi.tuna.tsinghua.edu.cn/simple/
#（4）中国科学技术大学     https://pypi.mirrors.ustc.edu.cn/simple/
#（5）华中科技大学  https://pypi.hustunique.com/
```



```bash
# 创建一个沙箱，python 大于等于3.8
conda create -n yolov5 python=3.10

conda activate yolov5
# 下载yolov5源代码库
git clone https://github.com/ultralytics/yolov5.git

cd yolov5

# 
pip install -r requirements.txt		# -U参数不用指定	
# -U：-U, --upgrade            Upgrade all specified packages to the newest available version. The handling of dependencies depends on the upgrade-strategy used.
# -r, --requirement <file>    Install from the given requirements file. This option can be used multiple times.

```



## 0.3 coco数据集

```bash
# coco
wget http://images.cocodataset.org/zips/train2017.zip	# 19G, 118k images
wget http://images.cocodataset.org/zips/val2017.zip		# 1G, 5k images
wget http://images.cocodataset.org/zips/test2017.zip	# 7G, 41k images
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip	# 数据的标签，解压上面的图片到此label文件夹内。

# coco128，从train2017随即选取的128张图片
https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip

# 下载yolov5对应代码的版本tag版本，在这里我们用的时tag v5.0版本
https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.
https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt
https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l.pt
https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.pt
```

# 1 全流程

## 1.1 标注数据

需要在有界面的主机上安装，远程ssh无法使用窗口

```bash
# 安装
pip install labelImg
# 启动
labelImg
```

标注

![img](./legend/wp.jpeg)

![img](./legend/wp-1703317059823-3.jpeg)

- 一张图片对应一个txt标注文件（如果图中无所要物体，则无需txt文件）；
- txt每行一个物体（一张图中可以有多个标注）；
- 每行数据格式：`类别id、x_center y_center width height`；
- **xywh**必须归一化（0-1），其中`x_center、width`除以图片宽度，`y_center、height`除以画面高度；
- 类别id必须从0开始计数。

## 1.2 准备数据集

### 数据集结构与存放位置

```bash
. 工作路径
├── datasets
│   └── person_data
│       ├── images
│       │   ├── train
│       │   │   └── demo_001.jpg
│       │   └── val
│       │       └── demo_002.jpg
│       └── labels
│           ├── train
│           │   └── demo_001.txt
│           └── val
│               └── demo_002.txt
└── yolov5
```

**要点：**

- `datasets`与`yolov5`同级目录；
- 图片 `datasets/person_data/images/train/{文件名}.jpg`对应的标注文件在 `datasets/person_data/labels/train/{文件名}.txt`，YOLO会根据这个映射关系自动寻找（`images`换成`labels`）；
- 训练集和验证集
  - `images`文件夹下有`train`和`val`文件夹，分别放置训练集和验证集图片;
  - `labels`文件夹有`train`和`val`文件夹，分别放置训练集和验证集标签(yolo格式）;

###  创建数据集的配置文件

复制`yolov5/data/coco128.yaml`一份，比如为`coco_person.yaml`

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/person_data  # 数据所在目录
train: images/train  # 训练集图片所在位置（相对于path）
val:  images/val # 验证集图片所在位置（相对于path）
test:  # 测试集图片所在位置（相对于path）（可选）

# 类别
nc: 5  # 类别数量
names: ['pedestrians','riders','partially-visible-person','ignore-regions','crowd'] # 类别标签名
```

### 选择并创建模型的配置文件



> 官方权重下载地址：https://github.com/ultralytics/yolov5

![img](./legend/wp-1703317810676-6.jpeg)

根据你的设备，选择合适的预训练模型，具体模型比对如下：

![img](./legend/wp-1703317810676-7.jpeg)

复制`models`下对应模型的`yaml`文件，重命名，比如课程另存为`yolov5s_person.yaml`，并修改其中：

```shell
# nc: 80  # 类别数量
nc: 5  # number of classes
```

### 训练

下载对应的预训练模型权重文件，可以放到`weights`目录下，设置本机最好性能的各个参数，即可开始训练，课程中训练了以下参数：

```shell
# yolov5s 
python ./train.py --data ./data/coco_person.yaml --cfg ./models/yolov5s_person.yaml --weights ./weights/yolov5s.pt --batch-size 32 --epochs 120 --workers 0 --name s_120 --project yolo_person_s
```

> 更多参数见`train.py`；
>
> 训练结果在`yolo_person_s/`中可见，一般训练时间在几个小时以上。



# log

1. [运行yolov5-5.0出现AttributeError: Can‘t get attribute ‘SPPF‘ 正确解决方法](https://blog.csdn.net/qq_41035097/article/details/122884652)

   - weight预置权重版本和实际yolov5的tag不匹配
   - 不能用weight 7.0 给yolov5 tagv5.0来训练

2. [AttributeError: module numpy has no attribute int .报错解决方案](https://blog.csdn.net/weixin_46669612/article/details/129624331)

   - 官方给出的numpy的版本要求时>=1.18.5，而[numpy](https://so.csdn.net/so/search?q=numpy&spm=1001.2101.3001.7020).int在[NumPy](https://so.csdn.net/so/search?q=NumPy&spm=1001.2101.3001.7020) 1.20中已弃用，在NumPy 1.24中已删除。
   - 重装numpy：pip install numpy==1.22

3. [RuntimeError: result type Float can‘t be cast to the desired output type long int](https://blog.csdn.net/bu_fo/article/details/130336910)

   ```python
   # loss.py出问题
   indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
   # 解决
   indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))  # image, anchor, grid indices
   ```

   

4. 



