# TensorRT

# 0 绪论

TensorRT是Nidia推出的深度学习推理SDK，能够在Nvidia GPU上实现低延迟，高吞吐的部署

**TensorRT包含用于训练好的模型优化器，以及用于执行推理的runtime（运行时环境）。**

我们可以将TensorRT看成是只有前向传播的深度学习框架，也就是说它只能用于推理。而我们常见的tensorflow，pytorch等等，这些框架是拿来训练的，它们训练好的模型，经过TensorRT的优化器优化，再经由runtime就可以在Nvidia的设备上进行高性能的推理。

![](./legend/TensorRT框架.png)

## 0.1 优化

TensorRT的核心在于对模型算子的优化，比如：合并算子，利用GPU的特性来选择特定和核函数等等。

![](./legend/优化策略.png)



![点击查看图片来源](./legend/422005c3c6c37c08f42897ebc162ece0.png)

TensorRT需要在目标GPU设备（根据硬件，软件环境版本）上实际运行来选择最优算法和配置，所以**TensorRT生成的模型迁移到别的设备或其他版本的TensorRT不一定能运行。**

## 0.2 模型转换

![点击查看图片来源](./legend/u=1090485776,1045116003&fm=253&fmt=auto&app=138&f=PNG.png)

方式1：通过TensorRT api手动定义网络，填充网络权重。

- 比较耗时，难以调试，通用性较差

方式2：直接从其他深度学习模型框架里导入，然后转换为TensorRT可以使用的模型文件。

由于深度学习发展很快，存在一些TensorRT不支持的自定义操作或算子层。TensorRT支持自定义插件的操作层，我们可以使用api来自定义那些TensorRT不支持的操作。

![img](./legend/u=421135204,1617536228&fm=253&app=138&f=JPEG.jpeg)

## 0.3 环境搭建

流程：

1. GPU驱动安装
2. Cuda 安装
3. cudnn 安装
4. TensorRT安装

```bash
# GPU驱动安装
# 检查机器建议的驱动
ubuntu-drivers devices

== /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
modalias : pci:v000010DEd000021C4sv00001462sd0000C75Abc03sc00i00
vendor   : NVIDIA Corporation
model    : TU116 [GeForce GTX 1660 SUPER]
driver   : nvidia-driver-470 - distro non-free
driver   : nvidia-driver-535-open - distro non-free
driver   : nvidia-driver-535-server - distro non-free
driver   : nvidia-driver-450-server - distro non-free
driver   : nvidia-driver-525 - distro non-free
driver   : nvidia-driver-535 - distro non-free recommended
driver   : nvidia-driver-525-server - distro non-free
driver   : nvidia-driver-535-server-open - distro non-free
driver   : nvidia-driver-470-server - distro non-free
driver   : nvidia-driver-525-open - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin

# 安装上面合适的驱动
apt install nvidia-driver-535

# 重启电脑
reboot

# 查看是否安装成功
nvidia-smi
```

## 0.4 tensorRT编程模型

模型转换的方式2

TensorRT分两个阶段运行

- 构建（`Build`）阶段：你向TensorRT提供一个模型定义，TensorRT为目标GPU优化这个模型。这个过程可以离线运行。
- 运行时（`Runtime`）阶段：你使用优化后的模型来运行推理。

构建阶段后，我们可以将优化后的模型保存为模型文件，模型文件可以用于后续加载，以省略模型构建和优化的过程。

### 0.4.1 构建阶段

构建阶段的最高级别接口是 `Builder`。`Builder`负责优化一个模型，并产生`Engine`。通过如下接口创建一个`Builder` 。

```
nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
```

要生成一个可以进行推理的`Engine`，一般需要以下三个步骤：

- 创建一个网络定义
- 填写`Builder`构建配置参数，告诉构建器应该如何优化模型
- 调用`Builder`生成`Engine`

# log

1. [CMP0104: CMAKE_CUDA_ARCHITECTURES now detected for NVCC, empty CUDA_ARCHITECTURES not allowed](https://blog.csdn.net/qq_33642342/article/details/116459742)

   ```cmake
   if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
       set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
   endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
   ```

   或者将cmake版本降低到3.17.1及以下就行

2. [Error Code 6: Internal Error (Unable to load library: libnvinfer_builder_resourc](https://blog.csdn.net/zlj1572043077/article/details/130466518)

   ```bash
   # 找到libnvinfer_builder_resource.so.8.6.0的位置，在把它复制到/usr/lib
   sudo cp ./libnvinfer_builder_resource.so.8.6.0  /usr/lib
   # 或者在/usr/lib里建立一个libnvinfer_builder_resource.so.8.6.0的软连接
   ln -s ./libnvinfer_builder_resource.so.8.6.0 /usr/lib/libnvinfer_builder_resource.so.8.6.0
   ```

   