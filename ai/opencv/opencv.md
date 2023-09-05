# OpenCV

# 0 基础

机器视觉与图形学之间的关系

opencv与ffmpeg



车辆检测项目

1. 窗口展示
2. 图像/视频的加载
3. 基本图形的绘制
4. 车辆的识别



## 基本图形的绘制

1. 色彩空间

   - RGB：人眼的色彩空间，OpenCV默认使用BGR，顺序不一样而已，主要用在物理显示器，器件当中

   - HSV/HSB/HSL：（主要用在OpenCV）

     - Hue（色相，色调，色彩，0°～360°）
     - Saturation（饱和度，颜色的纯度，0.0～1.0）
     - Value，Brightness，Lightness（亮度，0.0(黑色)～1.0(白色)）

     <img src="./legend/HSV.png" style="zoom:50%;" />

     <img src="./legend/YUV420.png" style="zoom:60%;" />

     YUV 4:2:0：在横向的4个像素里面，y在4个中都有，这里的2如果指U，那么V就是0，如果指V，那么U就是0

2. 像素访问

3. 矩阵基本运算

   - 创建矩阵array()，zeros()，ones()，full()，indentity（方形）/eye()
   - 检索和赋值
   - 获取子数组



OpenCV最重要的数据结构：Mat类表示的是一个n维的 稠密的 单通道或多通道的数值数组。



## 车辆识别涉及知识

1. 基本图像运算与处理
   - 矩阵四则运算，溶合（加权加法），位运算
   - 缩放，翻转，旋转
   - 仿射，透视
   - 图像滤波
2. 形态学
   - 
3. 轮廓查找

# 1 基本图像运算与处理

### [仿射变换](https://blog.csdn.net/u011681952/article/details/98942207)

#### 仿射变换数学表达

一个集合 XX 的仿射变换为：
$$
f(x) = Ax + b  \space\space\space x\in X
$$
仿射变换是二维平面中一种重要的变换，在图像图形领域有广泛的应用，在二维图像变换中，一般表达为：
$$
\left[
\begin{matrix}
x^\prime \\
y^\prime  \\
0  \\
\end{matrix}
\right]

=

\left[
\begin{matrix}
R_{00} & R_{01} & T_x   \\
R_{10} & R_{11} & T_y   \\
0 & 0 & 1  \\
\end{matrix}
\right]


\left[
\begin{matrix}
x\\
y\\
1 \\
\end{matrix}
\right]
$$
可以视为**线性变换R**和**平移变换T**的叠加

![](./legend/affine.png)

#### 平移变换矩阵

$$
M=
\left[
\begin{matrix}
1 & 0 & T_x   \\
0 & 1 & T_y   \\
0 & 0 & 1  \\
\end{matrix}
\right]
$$

#### 反射变换矩阵

$$
\left[
\begin{matrix}
1 & 0 & 0  \\
0 & -1 & 0  \\
0 & 0 & 1  \\
\end{matrix}
\right]
$$

#### API

```python
# 仿射变换
warpAffine(img, M, (w, h))
# 通过操作获取仿射变换矩阵
getRotationMatrix2D( center, angle, scale )
# 通过变换前多个点和变换后的多个对应的点（三个点），来获取仿射变换矩阵
getAffineTransform(src,dst)
```

### [透视变换](https://blog.csdn.net/bby1987/article/details/106317354)

<img src="./legend/perspective.png" style="zoom:50%;" />

### 图像滤波

一幅图像通过滤波器得到另一幅图像，其中滤波器又称卷积核，滤波的过程称为卷积

![](./legend/filtering.png)

概念：

- 卷积核的大小，3x3，5x5等，一般为奇数
  - 奇数的原因：一方面是增加padding的原因，一方面是保证锚点在中间，防止位置发生偏移的原因
  - 在深度学习中，卷积核越大，看到信息（感受野）越多，提取特征越好，同时计算量也就越大（多个小的卷积核去替代一个大的卷积核）
- 锚点
  - 卷积核的中心点
- 边界扩充
  - 当卷积核尺寸大于1，输出的尺寸会相应变小，为了是输入和输出的尺寸相等，则会对原始图像进行边界扩充，然后得到的输出尺寸就一致了。
  - `N =  (W - F + 2P) / S  + 1`，N输出图像的大小，W源图的大小，F卷积核大小，P为扩充尺寸，S为步长
- 步长
  - 卷积核每次偏移的大小

```python
filter2D(src, ddepth, kernel, anchor, delta, borderType)
# src 源图
# ddepth 位深，-1和源图保持一致
# kernel卷积核
# anchor -1根据卷积核找锚点
# delta 偏移
# borderType 边界的类型
```

1. 低通滤波，可以去除噪音或平滑图像（ps去痘）
   - 方盒滤波（boxFilter）和模糊滤波（blur）
   - 高斯滤波（GaussianBlur），卷积核锚点权重最高，越远离锚点权重越低。可以除掉图像中随机出现的**高斯噪声**
   - 中值滤波（medianBlur），取多个像素点中间的值，作为输出图像的像素值，对于**胡椒噪音有滤出**作用
   - 双边滤波（bilateralFilter），可以保留边缘，同时对边缘内的区域进行平滑处理
     - 高斯滤波之所以会导致图像变得模糊，是因为它在滤波过程中只关注了位置信息
2. 高通滤波，可以帮助查找图像的边缘
   - Sobel索贝尔，内部先使用高斯滤波（对噪音的适应性比较强），再求一阶导，
   - scharr沙尔，固定尺寸3x3，如果索贝尔的卷积核设为-1就约等于沙尔，但沙尔的效果要好一些，但索贝尔可以调整卷积核的大小
   - Laplacian拉普拉斯，**两个方向**可以同时求边缘，但它没有降噪，通常需要自己配合其他降噪使用
     - **沙尔和索贝尔一次只可以求一个方向的边缘，两个方向求完之后，再做加法**
   - [Canny](https://blog.csdn.net/m0_51402531/article/details/121066693)
     - 使用5x5高斯滤波消除噪声
     - 计算四个方向的梯度（0/45/90/135度）
     - 取梯度局部最大值，进行阈值（<min<grad<max<）计算，小于min为非边缘，大于max为强边缘，二者之间虚边缘再做区分
       - 与强边缘连接，则将该边缘处理为边缘
       - 与强边缘无连接，则该边缘为弱边缘，将其抑制。

# 2 形态学

定位物体的位置

形态学图像处理

- 腐蚀与膨胀
- 开运算（先腐蚀，后膨胀）
- 闭运算（先膨胀，后腐蚀）
- 顶帽 （源图 - 开运算）
- 黑帽（

这些是对二进制图像做处理的方法，也是一种卷积的做法。卷积核决定着图像处理后的效果



## 2.1 [图像二值化](https://zhuanlan.zhihu.com/p/360824614)

传统的机器视觉通常包括两个步骤：预处理和物体检测。而沟通二者的桥梁则是**图像分割（Image Segmentation）**[1]。图像分割通过简化或改变图像的表示形式，使得图像更易于分析。

最简单的图像分割方法是**二值化（Binarization）**。

- 全局阈值
- **自适应阈值**

```python
# 全局阈值
# threshold(img,thresh,maxval,type)，
# type：
# THRESH_BINARY 高于阈值设为maxval，低于阈值设为0 THRESH_BINARY_INV，与前面相反
# THRESH_TRUNC 高于阈值设为maxal，低于阈值保持源值
# THRESH_TOZERO 高于阈值保持不变，低于阈值设为0, THRESH_TOZERO_INV 高于阈值设为0，低于阈值保持源值
# 将源图转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二值化
ret, res = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

# 自适应阈值
# adaptiveThreshold(gray,maxValue,adaptiveMethod,type,blockSize,C)
res =cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
# adaptiveMethod：计算阈值的方法，ADAPTIVE_THRESH_MEAN_C计算邻近区域的平均值，ADAPTIVE_THRESH_GAUSSIAN_C高斯窗口加权平均值
# type：THRESH_BINARY，THRESH_BINARY_INV
# blockSize：光线影响较小可以设大一些
# C：常量，从计算的值中减去一个常量
```

自适应阈值，由于光照不均匀以及阴影的存在，全局阈值二值化会使得在阴影处的白色，被二值化成黑色

## 2.2 腐蚀和膨胀

**腐蚀和膨胀都是针对目标物来说的，目标物就是需要凸显的部分。**

**以下都是用黑底白字做实验，白色是需要凸显的部分**

腐蚀作用：消除物体的边界点，使边界向内收缩，可以把小于结构元素（核）的物体去除。可将两个有细小连通的物体分开。该方法可以用来去除毛刺，小凸起等。如果两个物体间有细小的连通，当结构足够大时，可以将两个物体分开。

当源图像与卷积核所有像素点一致时，才将源图上与锚点对应的点置为1，否则值为0。

```python
import cv2
import numpy as np

img = cv2.imread('../data/word.jpeg')        # 黑底白字
# 将源图转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel = np.ones((3,3),np.uint8)

# 获取卷积核
# cv2.getStructuringElement(type,size)
# type：
# MORPH_RECT 全一
# MORPH_ELLIPSE 椭圆形内是1,椭圆形外是0
# MORPH_CROSS 十字架上是1，十字架外是0
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7,7))
print(kernel)


res = cv2.erode(gray,kernel, iterations=1)       # iterations 腐蚀的次数


cv2.imshow('img', img)
cv2.imshow('gray',gray)
cv2.imshow('res', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

膨胀：卷积核的锚点对应的源像素点不为0，那就使源像素点周围的点都不为0

```python
cv2.dilate(gray, kernel, iterations=2)
```

噪点就是背景色（不需要突出的颜色）

思考：

1. **如果是白底黑字**，怎么腐蚀和膨胀
   - 黑底白字，腐蚀就是缩，膨胀就是胀
   - 白底黑字，腐蚀就是胀，膨胀就是缩
2. 卷积核是否可以设置为全0，可以为全0，处理前后图片没有变化

## 2.3 开运算和闭运算

- 开运算——先腐蚀，后膨胀，可以去除黑色背景中白色的噪点
- 闭运算——先膨胀，后腐蚀，可以去除白色需要凸显字体中的黑色噪点

```python
cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)     # 噪点越大，核选择就越大。MORPH_CLOSE
```

## 2.4 形态学梯度

梯度 = 源图 - 腐蚀的图。**用于求边缘**

```python
cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel) 
```

## 2.5 顶帽和黑帽

顶帽 = 源图 - 开运算，去除大目标，得到小目标物（得到黑色背景中的白色噪点）

```python
cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel) 
```

黑帽 = 源图 - 闭运算，得到白色目标物中的黑色噪点，再将黑色噪点转为白色

```python
cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
```



![](./legend/形态学处理.png)

# 3 轮廓

具有相同**颜色**或**强度**的**连续点**的曲线