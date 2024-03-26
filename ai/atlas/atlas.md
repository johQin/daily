# atlas 200

Atlas 200 AI加速模块是一款高性能的AI智能计算模块。

Atlas 200 AI加速模块集成了海思昇腾310 AI处理器（Ascend 310 AI处理器），可以实现图像、视频等多种数据分析与推理计算，可广泛用于智能监控、机器人、无人机、视频服务器等场景。

能力：

- 16TOPS INT8 @ 13W
- 16路高清视频实时分析，支持JPEG解码
- 8GB内存 | PCIe 3.0 x 4接口
- 工作温度：-25℃～+80℃
- 尺寸：52 x 38 x 10.2 mm

其中：

ascend 310 ai处理器 其中包括：2个DaVinci（达芬奇）AI Core，8个A55 Arm Core（最大主频1.6GHz）

# 0 绪论

## 0.1 [应用模式](https://support.huawei.com/enterprise/zh/doc/EDOC1100223190/9b43259d)

主处理器模式：Atlas 200 AI加速模块内部有8个Cortex-A55核，并提供常见的I2C、USB、SPI、RGMII等外设接口，可以作为嵌入式系统CPU使用。用户将操作系统烧录在eMMC Flash或SD卡中，经过简单的配置，可以让Atlas 200 AI加速模块中的ARM CPU运行用户指定的AI业务软件。

协处理器模式：Atlas 200 AI加速模块的ARM CPU仍然可以运行用户的AI业务软件。区别在于Atlas 200 AI加速模块作为协处理器时，系统中还存在一个主控CPU，Atlas 200 AI加速模块的外设接入、上电、休眠、唤醒等操作由主控CPU控制。用户的AI业务软件对外接口，也通过主控CPU上的软件转发。

从设备模式：Atlas 200 AI加速模块作为PCIe从设备接入CPU系统时，客户的AI业务程序运行在Host系统中，通过PCIe通道与Atlas 200 AI加速模块交互，将AI任务卸载到Ascend 310芯片中运行。

![](./legend/应用模式.png)

[以昇腾 AI 处理器的PCIe的工作模式进行区分](https://www.cnblogs.com/liqi175/p/16831761.html)：

RC模式 :  Root Complex

- 如果PCIe工作在主模式，可以扩展外设，则称为RC模式

EP模式:  EndPoint

- 如果PCIe工作在从模式，则称为EP模式。
- EP模式通常由Host侧作为主端，Device侧作为从端。客户的AI业务程序运行在Host系统中，产品作为Device系统以PCIe从设备接入Host系统，Host系统通过PCIe通道与Device系统交互，将AI任务加载到Device侧的昇腾 AI 处理器中运行。

![img](./legend/2939141-20221027115402053-1749146463.png)

Host和Device的概念说明如下：

- Host：是指与昇腾AI处理器所在硬件设备相连接的X86服务器、ARM服务器，利用昇腾AI处理器提供的NN（Neural-Network）计算能力完成业务。
- Device：是指安装了昇腾AI处理器的硬件设备，利用PCIe接口与服务器连接，为服务器提供NN计算能力。

理解：

- 如果把昇腾 AI 处理器所在的产品，当做一台电脑使用，自己可以处理所有东西，**独当一面。就是用RC模式**。
- 如果把昇腾 AI 处理器所在的产品，当做一个外设显卡使用，这时候需要跟别的主机配合使用，**团结合作，就是EP模式。**

[PCIe的RC模式和EP模式有什么区别？ ](https://www.cnblogs.com/yuanqiangfei/p/16649358.html)

- RC：Root Complex
  - RC设备用于连接CPU/内存子系统 和 I/O设备；
  - RC模式下，PCIE配置头中的类型值为1；
  - RC模式下，支持配置和I/O事务

- EP：EndPoint
  - EP设备通常表示一个串行或I/O设备；
  - EP模式下，PCIE配置头中的类型值为0；
  - EP模式下，PCIE控制器接收针对本地内存空间的读写操作

## 0.2 npu-smi

npu-smi是npu的系统管理工具，可以用于收集设备信息，查看设备健康状态，对设备进行配置以及执行固件升级、清除设备信息等功能。

驱动安装过程中会默认安装npu-smi工具。

对于Atlas 200 AI加速模块（EP模式）：安装完成后，npu-smi放置在“/usr/local/sbin/”和“/usr/local/bin/”路径下。

对于Atlas 200 AI加速模块（RC模式）：安装完成后，npu-smi放置在“/usr/local/sbin/”路径下。

不支持多线程并发使用npu-smi命令。

[npu-smi命令参考](https://support.huawei.com/enterprise/zh/doc/EDOC1100273887)

```bash
# 查询所有设备的基本信息
npu-smi info
+--------------------------------------------------------------------------------------------------------+
| npu-smi 22.0.4（npu-smi工具版本）                  Version: 22.0.4（npu驱动版本）                          |
+-------------------------------+-----------------+------------------------------------------------------+
| NPU     Name                  | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
| Chip    Device                | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
+===============================+=================+======================================================+
| 0       310                   | OK              | 12.8         51                0    / 187            |
| 0       0                     | NA              | 0            5745 / 7757                             |
+===============================+=================+======================================================+

字段说明
npu-sminpu-smi 工具版本
Version驱动版本
NPU设备ID
Name芯片名称，310——ascend 310芯片
Health芯片的健康状态，有五种状态：OK、Warning、Alarm、Critical或UNKNOWN
Power(W)芯片功率
Temp(C)芯片温度（单位°C）
Hugepages-Usage(page)大页占比（单位page），每一个page的大小是2048KB。
Chip芯片ID
Device芯片编号
Bus-IdBUS ID
AICore(%)AICore占用率
Memory-Usage(MB)内存占比

# 查询npu设备id
npu-smi info -l
	Card Count                     : 1

	NPU ID                         : 0
	Product Name                   : NA
	Serial Number                  : 033GWL10LB000840			# 产品序列号，可以在华为网站上注册，然后下载相关需要序列号的资料
	Chip Count                     : 1

# 用于查询设备的详细信息。-i后面的数字就是npu-smi info -l 显示的NPU ID
npu-smi info -t board -i 0
	NPU ID                         : 0
	Product Name                   : NA
	Model                          : NA
	Manufacturer                   : NA
	Serial Number                  : 033GWL10LB000840
	Software Version               : 22.0.4						# NPU驱动版本
	Firmware Version               : 1.84.15.1.310				# NPU固件版本
																# NPU驱动和固件包名称中包含的版本为6.0.RC1，但是部署驱动和固件后，
																# 使用npu-smi命令查询获取的驱动版本为22.0.3，固件版本为1.83.10.1.X。			
	Board ID                       : 0x3ec
	PCB ID                         : NA
	BOM ID                         : 0
	Chip Count                     : 1
	Faulty Chip Count              : 0

```

## 0.3 cann-toolkit & cann-nnrt

CANN（Compute Architecture for Neural Networks）是华为公司针对AI场景推出的异构计算架构，通过提供多层次的编程接口，支持用户快速构建基于昇腾平台的AI应用*和*业务。

Ascend-cann-toolkit 和 Ascend-cann-nnrt 是华为Atlas 200系列AI加速卡的软件开发工具包。

Ascend-cann-toolkit 主要用于模型开发和优化阶段，而 Ascend-cann-nnrt 则用于在Atlas 200设备上部署和执行训练好的深度学习模型。

1. **Ascend-cann-toolkit**:
   - 这个工具包通常用于开发人员和研究人员，提供了一系列的开发工具和库，以便他们可以利用Atlas 200系列硬件进行深度学习模型的开发和优化。
   - Ascend-cann-toolkit 提供了针对华为Ascend AI芯片的软件开发工具链，包括编译器、**模型转换工具**、性能分析工具等，使开发者可以在Atlas 200上开发和部署深度学习模型。
2. **Ascend-cann-nnrt**:
   - 这个工具包是华为提供的运行时环境，用于在Atlas 200设备上执行深度学习推理任务。
   - Ascend-cann-nnrt 提供了用于在Atlas 200设备上部署和执行深度学习模型的运行时库和工具，使用户可以在Atlas 200上实现高性能的推理任务。

## 0.4 [MindX生态](https://bbs.huaweicloud.com/blogs/281842)

从AI算法到产品化应用，有三大鸿沟：

1. 算法开发难：模型开发，模型训练，精度提高
2. 应用开发难：流解码，模型衔接，资源调度
3. 业务部署难：数据协同，设备协同，节点管理

为解决上述一系列的问题，华为推出了 Mindx 使能应用

Mindx 应用使能核心包含四大组件(2+1+X)：

- 2： Mindx DL 和 Mindx Edge：
  - Mindx DL 负责深度学习使能，核心是做集群计算（训练、推理） 
  - Mindx Edge 负责边云协同的组件。
- 1：ModelZoo 高性能模型库，选择在昇腾上效率和速率更高的模型 。
  - ModelZoo 为开发者提供多场景、高性能优选模型，筛选出对昇腾性能和效率较高的模型。
- X：Mindx SDK 面向 行业应用 SDK（开发套件）。
  - MindX SDK 每个模块就是一个插件；所有的插件都已经写好了，使用时 只需要替换om文件，就可以进行开发应用 。
  - 主要是为了快速完成 AI 应用开发，主要特点为：面向应用 API 封装、基础领域 SDK 、场景化方案 SDK、典型参考设计。
  - MindX SDK 提供基于流媒体框架 GStreamer 的流程编排与插件开发的开发方式，提供了流程编排 API ：10个，插件开发 API ：111个，相对于用 ApiSamples 进行开发 AI 应用大大缩减了代码开发量和缩短了业务上线是时间。 
  - 现阶段已发布2个 SDK 分别是：针对视频分析的 mxVision 和 针对智能制造的 mxManufacture。mxManufacture 现已在松山湖商用，mxVision 典型的应用场景为南瑞的电力检测等。

![](./legend/MindX生态.png)

## 0.5 [ascend 容器](https://ascendhub.huawei.cn/#/index)