# Deepstream

DeepStream是一个基于NVIDIA GPU和TensorRT的开源视频分析框架。

它提供了一个端到端的、可扩展的平台，可以处理多个视频和图像流，并支持实时的人脸识别、车辆识别、物体检测和跟踪、行为分析等视觉分析任务。DeepStream可以通过在不同的节点上进行分布式部署来实现高吞吐量和低延迟的处理，从而满足各种应用场景的需求，如智能城市、智能交通、工业自动化等。

Deepstream具备稳定高效的读流和推流能力；

![img](./legend/d58b851d78f348aeb78448e3e96ec5ab.png)

特性：

- 支持输入：USB/CSI 摄像头, 文件, RTSP流
- 示例代码：
  - C++: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_C_Sample_Apps.html
  - python: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Python_Sample_Apps.html
- 硬件加速插件：VIC, GPU, DLA, NVDEC, and NVENC
- 使用软件SDK： CUDA, TensorRT, NVIDIA® Triton™ （Deepstream将它们抽象为插件）
- 支持平台：Jetson , 各种 GPU, [container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream)

# 0 绪论

## 0.1 图架构（Graph architecture）

![img](legend/img202303271020872.png)

> 图示：
>
> - 典型的视频分析应用：都是从读取视频开始，从视频中抽取有价值信息，最后输出
> - 上半部分：用到的所有插件
> - 下半部分：整个App链路中用到的硬件引擎

图架构：

- Deepstream基于开源 [GStreamer](https://enpeicv.com/) 框架开发；
- 优化了内存管理：pipeline上插件之间没有内存拷贝，并且使用了各种加速器来保证最高性能；

插件（plugins）：

- input→ decode: [Gst-nvvideo4linux2](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvvideo4linux2.html)
- preprocessing:
  - [Gst-nvdewarper](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdewarper.html): 对鱼眼或360度相机的图像进行反扭曲
  - [Gst-nvvideoconvert](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvvideoconvert.html): 颜色格式的调整
  - [Gst-nvstreammux](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html): 多路复用器，从多个输入源形成一批缓冲区（帧）
  - inference:
    - [Gst-nvinfer](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html): TensorRT
    - [Gst-nvinferserver](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html): Triton inference server: native frameworks such as TensorFlow or PyTorch
  - [Gst-nvtracker](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html): 目标追踪
  - [Gst-nvdsosd](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdsosd.html): 可视化：bounding boxes, segment masks, labels
- output:
  - 格式： 窗口显示，保存到文件，流通过RTSP，发送元数据到云
  - [Gst-nvmsgconv](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvmsgconv.html) ：将元数据metadata转换为数据结构
  - [Gst-nvmsgbroker](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvmsgbroker.html)：向服务器发送遥测数据(如Kafka, MQTT, AMQP和Azure IoT)

## 0.2 应用架构（Application Architecture）

> https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html#reference-application-configuration

![img](./legend/应用架构.png)

- 包含了一系列的`GStreamer`插件
- 英伟达开发的插件有：
  - Gst-nvstreammux: 从多个输入源形成一批缓冲区（帧）
  - Gst-nvdspreprocess: 对预先定义的roi进行预处理，进行初步推理
  - Gst-nvinfer: TensorRT推理引擎（可用来检测和分类、分割）
  - Gst-nvtracker: 使用唯一ID来跟踪目标物体
  - Gst-nvmultistreamtiler: 拼接多个输入视频源显示
  - Gst-nvdsosd: 使用生成的元数据在合成视频帧上绘制检测框、矩形和文本等
  - Gst-nvmsgconv, Gst-nvmsgbroker: 将分析数据发送到云服务器。

## 0.3 环境准备

<table class="colwidths-given docutils align-default" id="id6">
<caption><span class="caption-text">dGPU model Platform and OS Compatibility</span><a class="headerlink" href="#id6" title="Permalink to this table"></a></caption>
<colgroup>
<col style="width: 17%">
<col style="width: 17%">
<col style="width: 17%">
<col style="width: 17%">
<col style="width: 17%">
<col style="width: 17%">
</colgroup>
<thead>
<tr class="row-odd"><th class="head"><p>DS release</p></th>
<th class="head"><p>DS 6.1</p></th>
<th class="head"><p>DS 6.1.1</p></th>
<th class="head"><p>DS 6.2</p></th>
<th class="head"><p>DS 6.3</p></th>
<th class="head"><p>DS 6.4</p></th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td><p>GPU platforms</p></td>
<td><p>T4, V100, A2, A10, A30, A100, RTX Ampere (Ax000/RTX30x0)</p></td>
<td><p>T4, V100, A2, A10, A30, A100, RTX Ampere (Ax000/RTX30x0)</p></td>
<td><p>T4, V100, A2, A10, A30, A100, RTX Ampere (Ax000/RTX30x0), Hopper, ADA</p></td>
<td><p>T4, V100, A2, A10, A30, A100, RTX Ampere (Ax000/RTX30x0), Hopper, ADA</p></td>
<td><p>T4, V100, A2, A10, A30, A100, RTX Ampere (Ax000/RTX30x0), Hopper, ADA</p></td>
</tr>
<tr class="row-odd"><td><p>OS</p></td>
<td><p>Ubuntu 20.04</p></td>
<td><p>Ubuntu 20.04</p></td>
<td><p>Ubuntu 20.04</p></td>
<td><p>Ubuntu 20.04</p></td>
<td><p>Ubuntu 22.04</p></td>
</tr>
<tr class="row-even"><td><p>GCC</p></td>
<td><p>GCC 9.4.0</p></td>
<td><p>GCC 9.4.0</p></td>
<td><p>GCC 9.4.0</p></td>
<td><p>GCC 9.4.0</p></td>
<td><p>GCC 11.4.0</p></td>
</tr>
<tr class="row-odd"><td><p>CUDA release</p></td>
<td><p>CUDA 11.6.1</p></td>
<td><p>CUDA 11.7.1</p></td>
<td><p>CUDA 11.8</p></td>
<td><p>CUDA 12.1</p></td>
<td><p>CUDA 12.2</p></td>
</tr>
<tr class="row-even"><td><p>cuDNN release</p></td>
<td><p>cuDNN 8.4.0.27</p></td>
<td><p>cuDNN 8.4.1.50+</p></td>
<td><p>cuDNN 8.7.0.84-1+</p></td>
<td><p>cuDNN 8.8.1.3-1+</p></td>
<td><p>cuDNN 8.9.4.25-1+</p></td>
</tr>
<tr class="row-odd"><td><p>TRT release</p></td>
<td><p>TRT 8.2.5.1</p></td>
<td><p>TRT 8.4.1.5</p></td>
<td><p>TRT 8.5.2.2</p></td>
<td><p>TRT 8.5.3.1</p></td>
<td><p>TRT 8.6.1.6</p></td>
</tr>
<tr class="row-even"><td><p>Display Driver</p></td>
<td><p>R510.47.03</p></td>
<td><p>R515.65.01</p></td>
<td><p>R525.85.12</p></td>
<td><p>R525.125.06</p></td>
<td><p>R535.104.12</p></td>
</tr>
<tr class="row-odd"><td><p>VideoSDK release</p></td>
<td><p>SDK 9.1</p></td>
<td><p>SDK 9.1</p></td>
<td><p>SDK 9.1</p></td>
<td><p>SDK 9.1</p></td>
<td><p>SDK 9.1</p></td>
</tr>
<tr class="row-even"><td><p>OFSDK release</p></td>
<td><p>2.0.23</p></td>
<td><p>2.0.23</p></td>
<td><p>2.0.23</p></td>
<td><p>2.0.23</p></td>
<td><p>2.0.23</p></td>
</tr>
<tr class="row-odd"><td><p>GStreamer release</p></td>
<td><p>GStreamer 1.16.2</p></td>
<td><p>GStreamer 1.16.2</p></td>
<td><p>GStreamer 1.16.3</p></td>
<td><p>GStreamer 1.16.3</p></td>
<td><p>GStreamer 1.20.3</p></td>
</tr>
<tr class="row-even"><td><p>OpenCV release</p></td>
<td><p>OpenCV 4.2.0</p></td>
<td><p>OpenCV 4.2.0</p></td>
<td><p>OpenCV 4.2.0</p></td>
<td><p>OpenCV 4.2.0</p></td>
<td><p>OpenCV 4.5.4</p></td>
</tr>
<tr class="row-odd"><td><p>Docker image</p></td>
<td><p>deepstream:6.1</p></td>
<td><p>deepstream:6.1.1</p></td>
<td><p>deepstream:6.2</p></td>
<td><p>deepstream:6.3</p></td>
<td><p>deepstream:6.4</p></td>
</tr>
<tr class="row-even"><td><p>NVAIE release</p></td>
<td><p>NA</p></td>
<td><p>NA</p></td>
<td><p>NVAIE-3.x</p></td>
<td><p>NVAIE-3.x</p></td>
<td><p>NVAIE-4.x</p></td>
</tr>
</tbody>
</table>

[在docker中准备环境](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html#a-docker-container-for-dgpu)：

[查找不同版本的deepstream的容器镜像](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream/tags)

```bash
docker pull nvcr.io/nvidia/deepstream:6.4-gc-triton-devel
# gc：gpu container
# triton：Triton Inference Server集成的版本
# "triton" 是指 NVIDIA Triton Inference Server（前身是TensorRT Inference Server）。Triton是一个用于部署深度学习模型的推理服务器，它提供了一个高性能、可扩展、多模型、多版本的推理服务平台。
# 具有以下主要特点：
#    多模型支持： Triton可以同时部署和管理多个深度学习模型，使得在同一服务器上可以运行不同任务的推理服务。
#    多版本支持： Triton支持在同一模型上部署多个不同版本，方便进行模型更新和回滚。
#    高性能： Triton通过与NVIDIA TensorRT等硬件加速库的集成，实现对深度学习推理任务的高性能加速。
#    容器化部署： Triton提供了基于Docker容器的部署方式，方便在容器化环境中进行推理服务的部署和管理。
#    RESTful API： Triton通过RESTful API提供推理服务，这使得客户端可以通过HTTP/HTTPS协议与服务器进行通信。

# 运行镜像
docker run -it -p 6522:22 -v /home/buntu/docker:/var/docker --gpus all nvcr.io/nvidia/deepstream:6.4-gc-triton-devel /bin/bash


```



# 1 运行

## 1.1 以配置文件的方式运行deepstream

### 1.1.3 配置文件解析

参考链接：https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html

```txt
# deepstream_app_config_windows_display_write.txt

[application]
enable-perf-measurement=1
perf-measurement-interval-sec=1

[tiled-display]
enable=0
rows=1
columns=1
width=1280
height=720
gpu-id=0
nvbuf-memory-type=0

[source0]
enable=1
type=3
uri=file:///home/enpei/Documents/course_cpp_tensorrt/course_10/1.deep_stream/media/sample_720p.mp4
num-sources=1
gpu-id=0
cudadec-memtype=0



[sink0]
enable=1
type=4
sync=0
gpu-id=0
codec=1
bitrate=5000000
rtsp-port=8554
udp-port=5400
nvbuf-memory-type=0


[sink1]
enable=1
type=2
sync=0
gpu-id=0
codec=1
width=1280
height=720
nvbuf-memory-type=0

[sink2]
enable=1
type=3
sync=1
gpu-id=0
codec=1
container=1
bitrate=5000000
output-file=/home/enpei/Documents/course_cpp_tensorrt/course_10/1.deep_stream/media/output.mp4
nvbuf-memory-type=0

[osd]
enable=1
gpu-id=0
border-width=5
border-color=0;1;0;1
text-size=30
text-color=1;1;1;1;
text-bg-color=0.3;0.3;0.3;1
font=Serif
show-clock=0
clock-x-offset=800
clock-y-offset=820
clock-text-size=12
clock-color=1;0;0;0
nvbuf-memory-type=0

[streammux]
gpu-id=0
live-source=0
batch-size=1
batched-push-timeout=40000
width=1920
height=1080
enable-padding=0
nvbuf-memory-type=0

[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=config_infer_primary_yoloV5.txt

[tests]
file-loop=1

```

![img](./legend/wp.jpeg)

**[application]**

- `enable-perf-measurement=1`：启用性能测量。
- `perf-measurement-interval-sec=5`：性能测量间隔。

**[tiled-display]**

> 多个输入源显示在同一个窗口中

- `enable=0`：是否启用Tiled Display。
- `rows=1`：Tiled Display中的行数。
- `columns=1`：Tiled Display中的列数。
- `width=1280`：Tiled Display中每个小窗口的宽度。
- `height=720`：Tiled Display中每个小窗口的高度。
- `gpu-id=0`：Tiled Display所在的GPU ID。
- `nvbuf-memory-type=0`：NvBufSurface的内存类型。

**源属性 source**

> Deepstream支持输入多个输入源，对于每个源需要单独命名为`source%d`，并且添加相应配置信息，类似：

```txt
[source0]
key1=value1
key2=value2
...

[source1]
key1=value1
key2=value2
...

```

- `enable=1`：是否启用源。

- `type=3`：源类型，
  - 1: Camera (V4L2)
  - 2: URI（统一资源标识符）
  - 3: MultiURI
  - 4: RTSP
  - 5: Camera (CSI) (Jetson only)
- `uri`：媒体文件的URI，可以是文件（`file:///app/2.ds_tracker/media/sample_720p.mp4`），也可以是`http, RTSP `流。
- `num-sources=1`：源数量。
- `gpu-id=0`：源所在的GPU ID。
- `cudadec-memtype=0`：解码器使用的内存类型。

**sink**

> sink组件用于控制显示的渲染、编码、文件存储，pipeline可以支持多个sink，命名为`[sink0],[sink1]...`

- `enable=1`：是否启用。

- `type=4`：类型，4表示RTSP流，更多类型如下：
  - 1: Fakesink
  - 2: EGL based windowed nveglglessink for dGPU and nv3dsink for Jetson （窗口显示）
  - 3: Encode + File Save (encoder + muxer + filesink) （文件）
  - 4: Encode + RTSP streaming; Note: sync=1 for this type is not applicable; （推流）
  - 5: nvdrmvideosink (Jetson only)
  - 6: Message converter + Message broker
- `sync=0`：流渲染速度，0: As fast as possible  1: Synchronously
- `gpu-id=0`：汇所在的GPU ID。
- `codec=1`：编码器类型，1: H.264 (hardware)，2: H.265 (hardware)
- `bitrate=5000000`：比特率。
- `rtsp-port=8554`：RTSP端口号。
- `udp-port=5400`：UDP端口号。
- `nvbuf-memory-type=0`：NvBufSurface的内存类型。

**osd**

> 启用On-Screen Display (OSD) 屏幕绘制

- `enable=1`：是否启用OSD。
- `gpu-id=0`：OSD所在的GPU ID。
- `border-width=5`：检测框宽度（像素值）。
- `border-color=0;1;0;1`：检测框颜色（R;G;B;A Float, 0≤R,G,B,A≤1）。
- `text-size=15`：文本大小。
- `text-color=1;1;1;1;`：文本颜色。
- `text-bg-color=0.3;0.3;0.3;1`：文本背景颜色。
- `font=Serif`：字体。
- `show-clock=1`：是否显示时钟。
- `clock-x-offset=800`：时钟的X方向偏移量。
- `clock-y-offset=820`：时钟的Y方向偏移量。
- `clock-text-size=12`：时钟文本大小。
- `clock-color=1;0;0;0`：时钟文本颜色。
- `nvbuf-memory-type=0`：NvBufSurface的内存类型。

**Streammux属性**

> 设置流多路复用器

- `gpu-id=0`：Streammux所在的GPU ID。
- `live-source=0`：是否使用实时源。
- `batch-size=1`：批大小。
- `batched-push-timeout=40000`：批处理推送超时时间。
- `width=1920`：流的宽度。
- `height=1080`：流的高度。
- `enable-padding=0`：是否启用填充。
- `nvbuf-memory-type=0`：NvBufSurface的内存类型。

**Primary GIE属性**

> GIE （GPU Inference Engines）图像推理引擎，支持1个主引擎和多个次引擎，例如：

```txt
[primary-gie]
key1=value1
key2=value2
...

[secondary-gie1]
key1=value1
key2=value2
...

[secondary-gie2]
key1=value1
key2=value2
...

```

- `enable=1`：是否启用Primary GIE。
- `gpu-id=0`： GIE所在的GPU ID。
- `gie-unique-id=1`： GIE的唯一ID。
- `nvbuf-memory-type=0`：NvBufSurface的内存类型。
- `config-file=config_infer_primary_yoloV5.txt`：引擎的配置文件路径，参考后文介绍，可以包含 [这个表格](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html#primary-gie-and-secondary-gie-group) 所有属性（除`config-file`）

- **Tests属性**

  > 用于调试

  `file-loop=1`：是否启用文件循环。

对应我们也需要对gie进行配置，配置文件为`config_infer_primary_yoloV5.txt`

```txt
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file=../weights/yolov5s.onnx
model-engine-file=../weights/yolov5.engine
infer-dims=3;640;640
labelfile-path=labels.txt
batch-size=1
workspace-size=1024
network-mode=2
num-detected-classes=80
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path=../lib/yolov5_decode.so

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300

```

每个参数的详细解释：

> 原始参考链接：https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html#gst-nvinfer-file-configuration-specifications

- [property]：指定GIE的各种属性和参数：
  - `gpu-id=0`：使用的GPU ID。
  - `net-scale-factor=0.0039215697906911373`：输入图像归一化因子，为1/255。
  - `model-color-format=0`：输入图像颜色格式，0: RGB 1: BGR 2: GRAY
  - `onnx-file=yolov5s.onnx`：输入的ONNX模型文件路径（如果`model-engine-file`指向的文件不存在，则会用onnx 生成一个，并且结合`network-mode`调整精度）
  - `model-engine-file=yolov5.engine`：TensorRT模型engine文件路径（如果指向文件不存在，参考上一步，如果存在，`network-mode`不起作用）
  - `infer-dims=3;640;640`：输入图像的维度，格式为`通道数;宽度;高度`。
  - `labelfile-path=labels.txt`：类别标签文件路径。
  - `batch-size=1`：推理时的批大小。
  - `workspace-size=1024`：TensorRT内部使用的工作空间大小。
  - `network-mode=2`：推理模式，0: FP32 1: INT8 2: FP16
  - `num-detected-classes=80`：模型可检测的类别数。
  - `interval=0`：推理时图像的采样间隔，0表示不采样。
  - `gie-unique-id=1`：TensorRT引擎的唯一ID。
  - `process-mode=1`：推理时使用的处理模式，1表示同步处理。
  - `network-type=0`：**神经网络类型，0: Detector，1: Classifier，2: Segmentation，3: Instance Segmentation**
  - `cluster-mode=2`：推理时的集群模式，0: OpenCV groupRectangles() 1:  DBSCAN 2: Non Maximum Suppression 3: DBSCAN + NMS Hybrid 4: No  clustering （for instance segmentation）
  - `maintain-aspect-ratio=1`：是否保持输入图像的宽高比，1表示保持。
  - `parse-bbox-func-name=NvDsInferParseYolo`：解析边界框的函数名（这里用的是NvDsInferParseYolo）
  - `custom-lib-path=yolov5_decode.so`：自定义解码库路径（使用了一个自定义库文件yolov5_decode.so来解码推理输出。）
- [class-attrs-all]：指定目标检测的属性和参数：
  - `nms-iou-threshold=0.45`：非极大值抑制的IOU阈值，低于阈值的候选框会丢弃
  - `pre-cluster-threshold=0.25`：聚类前的阈值。
  - `topk=300`：按置信度排序，保留前K个目标。

## 1.2 以代码方式运行deepstream

### 1.2.1 Metadata

> - https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_metadata.html
> - https://docs.nvidia.com/metropolis/deepstream/dev-guide/sdk-api/index.html（优先）

核心数据：

- NvDsBatchMeta：`Gst-nvstreammux`处理后的批次数据
- 次级结构：
  - frame
  - object
  - classifier
  - label data

### 1.2.2 GStreamer 核心概念

> gstreamer是一种基于流的多媒体框架，可用于创建、处理和播放音频和视频流。它是一个开源项目，可以在Linux、Windows、macOS等多个平台上使用。gstreamer提供了一系列的库和插件，使开发者可以构建自定义的流媒体应用程序，包括音频和视频编辑器、流媒体服务器、网络摄像头应用程序等等。gstreamer具有高度的可扩展性和灵活性，可以通过插件的方式支持各种不同的编解码器、协议和设备。

#### 3.3.1 GstElement

> https://gstreamer.freedesktop.org/documentation/application-development/basics/elements.html?gi-language=c

- 媒体应用pipeline基本的构建模块（可以看成一个有输入输出的黑箱）

  - 输入：编码数据
  - 输出：解码数据

- 类型：

  - source element：没有输入，不接受数据，只产生数据（比如从文件中读取视频）

    ![img](legend/img202303271521713.png)

  - Filters, convertors, demuxers（分流器）, muxers（混流器） and codecs：可以有多个输入输出（source pad、sink pad）

    ![img](legend/img202303271521950.png)![img](legend/img202303271521947.png)

    - Sink pad: 接受，消费数据（图左）
    - Source pad：输出，生产数据（图右）

  - sink element：没有输出，不生产数据，只消费数据（比如写入磁盘、推流）

    ![img](legend/img202303271525358.png)

- 创建元素：

  ```c++
  GstElement *element;
  
  /* init GStreamer */
  gst_init (&argc, &argv);
  
  /* create element */
  element = gst_element_factory_make ("fakesrc", "source");
  if (!element) {
    g_print ("Failed to create element of type 'fakesrc'\n");
    return -1;
  }
  // unref 
  gst_object_unref (GST_OBJECT (element));
  ```

- 获取元素属性、设置元素属性、查询元素属性

  ```c++
  // get properties
  g_object_get()
  // set properties
  g_object_set()
  
  // query properties (bash command)
  gst-inspect element
  ```

- 常见元素：

  - filesrc
  - h264parse
  - nv412decoder
  - nvstreammux
  - nvinfer
  - nvtracker
  - nvvideoconvert
  - nvdsosd

- 链接元素

  ![img](legend/img202303271527901.png)

  ```c++
  GstElement *pipeline;
  GstElement *source, *filter, *sink;
  
  /* create pipeline */
  pipeline = gst_pipeline_new ("my-pipeline");
  
  /* create elements */
  source = gst_element_factory_make ("fakesrc", "source");
  filter = gst_element_factory_make ("identity", "filter");
  sink = gst_element_factory_make ("fakesink", "sink");
  
  /* must add elements to pipeline before linking them */
  gst_bin_add_many (GST_BIN (pipeline), source, filter, sink, NULL);
  
  
  // more specific behavior 
  gst_element_link () and gst_element_link_pads ()
  ```

- 元素状态：

  - GST_STATE_NULL

  - GST_STATE_READY

  - GST_STATE_PAUSED

  - GST_STATE_PLAYING

  - 设置函数：

    ```c++
    /* Set the pipeline to "playing" state */
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    ```

#### 3.3.2 bin

> https://gstreamer.freedesktop.org/documentation/application-development/basics/bins.html?gi-language=c

![img](legend/wp-1708503039802-6.jpeg)

- 一串链接起来的元素的集合（bin其实一种element）

- 创建方式：

  ```C++
  /* create */
  pipeline = gst_pipeline_new ("my_pipeline");
  bin = gst_bin_new ("my_bin");
  source = gst_element_factory_make ("fakesrc", "source");
  sink = gst_element_factory_make ("fakesink", "sink");
  
  /* First add the elements to the bin */
  gst_bin_add_many (GST_BIN (bin), source, sink, NULL);
  /* add the bin to the pipeline */
  gst_bin_add (GST_BIN (pipeline), bin);
  
  /* link the elements */
  gst_element_link (source, sink);
  ```

#### 3.3.3 pipeline

![img](legend/wp-1708503039802-7.jpeg)

- 所有的元素都必须添加进pipeline后才能被使用（pipeline需要调整时钟、消息传递等）

- pipeline也是一种特殊的bin，所以也是element

- 创建方式

  ```C++
  GstElement *pipeline;
  // create 
  pipeline = gst_pipeline_new (const gchar * name)
  
  // add elements to the pipeline
  gst_bin_add_many (GST_BIN (pipeline), source, sink, NULL);
  ```

#### 3.3.4 pads

> [https://gstreamer.freedesktop.org/documentation/application-development/basics/pads.html?gi-language=chttps://gstreamer.freedesktop.org/documentation/application-development/basics/pads.html?gi-language=c](https://gstreamer.freedesktop.org/documentation/application-development/basics/pads.html?gi-language=c)

![img](legend/wp-1708503039802-8.jpeg)

- pads是元素对外服务的接口
- 数据从一个元素的source pad流向下一个元素的sink pad。

### 3.4 课程应用

> 代码：4.ds_tracker

在课程的例子中，我们主要通过构建了整条pipeline，并传入gie以及nvtracker的配置，以搭建了一条满足我们应用的流水线。

其中`gie`的配置文件类似上文配置文件介绍：

```bash
# pgie_config.txt
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file=yolov5s.onnx
model-engine-file=../weights/yolov5.engine
labelfile-path=../weights/labels.txt
infer-dims=3;640;640
batch-size=1
workspace-size=1024
network-mode=2
num-detected-classes=80
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path=../build/libyolo_decode.so

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300
```

`nvtracker`的配置文件也类似：

```
[tracker]
tracker-width=640
tracker-height=384
gpu-id=0
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
ll-config-file=./config_tracker_IOU.yml
enable-batch-process=1
```

在docker容器内运行，示例如下：

- 仅检测

  pipeline流程：

  ![img](./legend/wp-1708503039802-9.jpeg)
