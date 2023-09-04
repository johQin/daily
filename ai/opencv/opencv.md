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



## 车辆识别

1. 基本图像运算与处理
   - 矩阵四则运算，溶合（加权加法），位运算
   - 缩放，翻转，旋转
   - 仿射，透视
   - 图像滤波
2. 形态学
3. 轮廓查找

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
   - 高斯滤波（GaussianBlur），卷积核锚点权重最高，越远离锚点权重越低
2. 高通滤波，可以帮助查找图像的边缘

