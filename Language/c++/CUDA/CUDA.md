# CUDA

[CUDA 编程教程推荐](https://zhuanlan.zhihu.com/p/346910129)



# 0 环境搭建

```bash
# 查看电脑的显卡
lspci | grep -i vga
```

## 0.1 安装问题纪实

[nvidia 显卡驱动 安装最顺的教程](https://zhuanlan.zhihu.com/p/302692454)

[选择显卡驱动版本和toolkit版本下载，不含安装报错的显卡驱动安装教程](https://blog.csdn.net/weixin_39928010/article/details/131142603)

```bash
# 1. 安装驱动，下载local(run file)

# Error : your appear to running an x server；please exit x before installing .for further details
# 解决方案： https://blog.csdn.net/qq_32415217/article/details/123185645
sudo chmod +x NVIDIA-Linux-x86_64-535.54.03.run
sudo ./NVIDIA-Linux-x86_64-535.54.03.run -no-x-check

# ERROR: The Nouveau kernel driver is currently in use by your system. This driver is incompatible with the NVIDIA driver……
sudo vi /etc/modprobe.d/nvidia-installer-disable-nouveau.conf
# 在文件尾部添加，没有禁用过nouveau，是没有这个文件的。
blacklist nouveau
options nouveau modeset=0
# 更新上述修改
sudo update-initramfs -u
# 重启电脑，记得一定要重启电脑
reboot

# 2. 安装cuda toolkit
sudo chmod +x cuda_12.0.0_525.60.13_linux.run
sudo ./cuda_12.0.0_525.60.13_linux.run
# 提前安装了驱动，在cuda toolkit 中就不要安装gpu驱动

sudo gedit ~/.bashrc
# 添加两个环境变量
export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# 更新环境
source ~/.bashrc

# 测试是否安装成功
nvcc -V

# 3. 跑cuda sample 代码
# 下载sample代码。cuda toolkit安装包在11.6之后便不再安装sample代码，需要自行在github上下载
git clone -b v12.0 --depth=1 git@github.com:NVIDIA/cuda-samples.git

# gpu设备查询sample
cd cuda-samples/Samples/1_Utilities/deviceQuery

make
# make 执行，生成的可执行文件放在cuda-samples/bin/x86_64/linux/release
/usr/local/cuda/bin/nvcc -ccbin g++ -I../../../Common -m64 --threads 0 --std=c++11 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o deviceQuery.o -c deviceQuery.cpp
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o deviceQuery deviceQuery.o 
mkdir -p ../../../bin/x86_64/linux/release
cp deviceQuery ../../../bin/x86_64/linux/release

# 这个文件夹下生成可执行文件deviceQuery
cd cuda-samples/bin/x86_64/linux/release
# 运行
./deviceQuery


./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 3060 Ti"
  CUDA Driver Version / Runtime Version          12.2 / 12.0
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 7972 MBytes (8359641088 bytes)
  (038) Multiprocessors, (128) CUDA Cores/MP:    4864 CUDA Cores
  GPU Max Clock rate:                            1695 MHz (1.70 GHz)
  Memory Clock rate:                             7001 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 3145728 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 3 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.2, CUDA Runtime Version = 12.0, NumDevs = 1
Result = PASS
```



FLOPS——float-point Operation per Second，每秒浮点操作次数，GFLOPS——g（billion） FLOPS，TFLOPS—— T（1000g） FLOPS

## 0.2 [初识和相关概念](https://zhuanlan.zhihu.com/p/34587739)

### 0.2.1 前言

GPU并不是一个独立运行的计算平台，而需要与CPU协同工作，可以看成是CPU的协处理器，因此当我们在说GPU并行计算时，其实是指的基于CPU+GPU的异构计算架构。

在异构计算架构中，GPU与CPU通过PCIe总线连接在一起来协同工作，CPU所在位置称为为主机端（host），而GPU所在位置称为设备端（device）

![](./legend/异构模型.png)

GPU包括更多的运算核心，其特别适合数据并行的计算密集型任务，如大型矩阵运算。

CPU的运算核心较少，但是其可以实现复杂的逻辑运算，因此其适合控制密集型任务。

另外，CPU上的线程是重量级的，上下文切换开销大，但是GPU由于存在很多核心，其线程是轻量级的。

因此，基于CPU+GPU的异构计算平台可以优势互补，CPU负责处理逻辑复杂的串行程序，而GPU重点处理数据密集型的并行计算程序，从而发挥最大功效。

![](./legend/cpu-gpu在程序中.webp)

CUDA是NVIDIA公司所开发的GPU编程模型，它提供了GPU编程的简易接口，基于CUDA编程可以构建基于GPU计算的应用程序。

CUDA提供了对其它编程语言的支持，如C/C++，Python，Fortran等语言

### 0.2.2 CUDA编程模型基础

CUDA编程模型是一个异构模型，需要CPU和GPU协同工作。

#### 概念：**host**和**device**

- host指代CPU及其内存
- device指代GPU及其内存。
- CUDA程序中既包含host程序，又包含device程序，它们分别在CPU和GPU上运行。同时，host与device之间可以进行通信，这样它们之间可以进行数据拷贝。
- 典型的CUDA程序的执行流程如下：
  1. 分配host内存，并进行数据初始化；

1. 分配device内存，并从host将数据拷贝到device上；
2. 调用CUDA的核函数在device上完成指定的运算；
3. 将device上的运算结果拷贝到host上；
4. 释放device和host上分配的内存



#### **kernel**

- kernel是在device上线程中并行执行的函数
- 函数类型区分：
  - `__global__`：
    - 在device上执行，从host中调用（一些特定的GPU也可以从device上调用）
    - 返回类型必须是`void`，不支持可变参数参数，不能成为类成员函数。
    - `__global__`定义的kernel是**异步**的，这意味着host不会等待kernel执行完就执行下一步。
  - `__device__`：在device上执行，单仅可以从device中调用
  - `__host__`：在host上执行，仅可以从host上调用，一般省略不写。可和`__device__`同时用，此时函数会在device和host都编译。



#### 线程结构的**层次**概念

- 第一层次网格grid：
  - kernel在device上执行时实际上是启动很多线程，一个kernel所启动的所有线程称为一个**网格**（grid），同一个网格上的线程共享相同的全局内存空间
- 第二层次线程块block：
  - 网格又可以分为很多**线程块**（block），线程块又包含许多线程。
- 第三层次线程thread
- 一个线程需要两个内置的坐标变量（blockIdx，threadIdx）来唯一标识，它们都是`dim3`类型变量，其中blockIdx指明线程所在grid中的位置，而threaIdx指明线程所在block中的位置

![](./legend/线程的层次结构.png)

```c++
// Kernel定义
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) 
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i < N && j < N) 
        C[i][j] = A[i][j] + B[i][j]; 
}
int main() 
{ 
    ...
    // Kernel 线程配置
    dim3 threadsPerBlock(16, 16); 
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    // kernel调用
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C); 
    ...
}
```



#### 内存模型

每个线程有自己的私有本地内存（Local Memory）

而每个线程块有包含共享内存（Shared Memory），可以被线程块中所有线程共享，其生命周期与线程块一致。

此外，所有的线程都可以访问全局内存（Global Memory）。还可以访问一些只读内存块：常量内存（Constant Memory）和纹理内存（Texture Memory）。

![](./legend/内存模型.webp)
