# STM32

STM32F103xE 性能线路块图表

![](./legend/STM32F103xE性能线路块图表.png)

# 0 绪论

## 0.1 介绍

ST指意法半导体（公司），M指微处理器，32表示计算机处理器位数（51单片机是8位的）。

ARM分成三个系列：

- Cortex-A：针对多媒体应用（手机平板，对性能要求比较高）
- Cortex-R：针对对实时性和性能有一定要求的场景
- Cortex-M：针对低功耗高性能的场景

stm32它是一个soc（System On Chips，片上集成系统），它集成很多模块。其中最重要的就是cpu核心，它的核心选择的就是Cortex-M。

ARM采用的是一个精简指令集。

### 0.1.0 STM32 与 Linux 系统的比较

STM32 作为一个微控制器，与基于 Linux 的系统在设计和应用方面存在明显差异。STM32 主要用于直接控制硬件和执行特定任务的嵌入式应用，而 Linux 系统通常用于更复杂的计算任务，需要较大的处理能力和内存。

在操作系统方面，STM32 通常**不运行传统的操作系统，而是执行裸机代码或实时操作系统（RTOS）**。这使得 STM32 在响应时间和资源占用上更高效，适合实时性要求高的应用。相比之下，Linux 是一个完整的操作系统，提供了丰富的功能和服务，但也因此需要更多的资源，如处理器性能和内存。

从应用角度来看，STM32 适用于需要精确时间控制和资源约束的场合，如传感器数据采集、电机控制等。而 Linux 系统更适合需要复杂数据处理、网络通信和用户交互的应用，如服务器、桌面计算等。

### 0.1.1 stm32系列

stm32根据闪存容量，引脚数量等构成了一个庞大的stm32家族（型号众多）。

命名规则：

![](./legend/31276c8225cd41f592c9a8c3110795b8.png)

F：通用快闪（FlashMemory）；
L：低电压（1.65～3.6V）；
F类型中F0xx和F1xx系列为2.0～3.6V;F2xx和F4xx系列为1.8～3.6V

### 0.1.2 stm32芯片内部结构

<img src="./legend/image-20240203143335925.png" alt="image-20240203143335925" style="zoom: 33%;" />

<img src="./legend/stm32微控制器内核.png" style="zoom: 50%;" />

DMA（Direct Memory Access）：它能使数据从附加设备（如磁盘驱动器）直接发送到计算机主板的内存上。

SRAM：随机存储器

APB（Advanced Peripheral Bus，高级外围设备总线）

FLASH：闪存，相当于硬盘，编写的代码都放这里面。

### 0.1.3 stm32最小系统

这一款是stm32型号：stm32f103c8t6

1. 供电系统
2. 时钟电路（内部时钟源、外部时钟源）
3. 复位电路
4. 下载的接口电路
5. boot电路

<img src="./legend/stm32最小系统.png" style="zoom:30%;" />

## 0.2 STM32F103ZET6 开发板

开发板的核心由 STM32F103ZET6 微控制器构成，它被各类接口和模块环绕，旨在提供一站式的开发体验。

![image-20240330145001236](./legend/image-20240330145001236.png)

## 0.3 开发方式

#### 语言

在开发 STM32 单片机项目时，开发者通常使用汇编语言和 C 语言这两种主要编程语言。汇编语言允许开发者直接与硬件进行交互，而 C 语言则提供了更高级、更易于管理的编程接口。

#### 操作方式

实际的编程过程中，有两种主要的方法来操作和控制 STM32 单片机：

1. 直接配置和操作微控制器的功能模块寄存器:
   - 这种方法涉及直接访问和修改微控制器内部的寄存器，以配置其硬件功能。
   - 它需要对 STM32 的硬件结构和寄存器映射有深入的了解，适合对性能和资源使用有特别要求的场合。
2. 使用 ST 官方提供的固件库和驱动进行操作：
   - ST 公司为 STM32 提供了官方的固件库，这些库封装了对硬件的操作，简化了编程过程。
   - 通过使用这些固件库，开发者可以更容易地实现对微控制器功能的控制，而不必深入到底层的硬件细节。

## 0.4 固件库

在 STM32 微控制器的开发领域中，有多种固件库可供选择，每种库都有其独特的特点和优势。

这些库包括最初的 Standard Peripheral Library（SPL）、更为现代化的 STM32Cube（包括 HAL 和 LL 库），以及广泛适用于 ARM Cortex-M 微控制器的 CMSIS。

- Standard Peripheral Library (SPL，标准外围设备库): 
  - 这是 ST Microelectronics 最初为其 STM32 微控制器系列发布的固件库。
  - 此库包含了一些方便的 C 函数，可以直接控制 STM32 的各种外设，通常称为标准库。
- STM32Cube
  - ST Microelectronics 自 2015 年以来开始推广的一种新的固件库
  - 包括一个嵌入式软件平台和一个独立的集成开发环境。
    - 嵌入式软件平台包括一个硬件抽象层(HAL，Hardware Abstract Layer )，该层为 STM32 的各种外设提供通用的 API，并且还包含一些中间件组件（如 FreeRTOS，USB 库，TCP/IP 库等）。
    - 集成开发环境（STM32CubeIDE）则包含了代码生成器，它可以生成基于STM32Cube HAL 的初始化代码。
- LL (Low Layer) Drivers
  - 是 STM32Cube 库的一部分，为高级用户提供了一个硬件抽象层的替代方案。LL 库提供了一组低级 API，可以让用户直接访问 STM32外设的寄存器。这些 **LL 的API 比 HAL 更加高效，但是需要更深入的硬件知识。**
- CMSIS (Cortex Microcontroller Software Interface Standard)
  - CMSIS 并不是一个STM32 特定的固件库，而是 ARM 公司为 Cortex-M 微控制器定义的一组接口。许
    多 STM32 固件库（包括 SPL 和 STM32Cube）都使用 CMSIS 作为底层的硬件抽象。

STM32Cube HAL 是 STMicroelectronics 为了替代 SPL 而开发的更现代、更全面的固件库。

它不仅提供了广泛的硬件支持，还包含了多种中间件组件，如 FreeRTOS、USB 库、TCP/IP 库等。HAL 库通过提供通用的 API，使得操作 STM32 的各种外设变得更加简单。这种抽象层降低了学习曲线，尤其适合初学者和那些希望快速实现项目原型的开发者。

STM32CubeIDE 集成了代码生成器，可以自动生成基于 HAL的初始化代码。这极大地简化了项目的初始配置过程，使开发者能够更快地进入实际的应用开发阶段。

HAL库学完，进阶可以再去学LL库，抛弃掉标准库（它学习曲线太陡峭了）。

## 0.5 开发环境

#### [工具](https://blog.csdn.net/weixin_45880844/article/details/133930223)

STM32CubeMX和Keil uVision5是两个不同的软件工具，但它们可以一起使用来开发STM32微控制器的应用程序。

##### STM32CubeMX

STM32CubeMX是STMicroelectronics提供的一款图形化配置工具，用于生成初始化代码和配置文件，文件后缀通常是“.c”和“.h”，这些文件是C语言源代码文件，用于配置和控制STM32微控制器的各种硬件特性和外设。它可以帮助程序员根据需求选择微控制器型号，并配置其硬件接口，如GPIO、ADC、DAC、UART、SPI等，并生成用于KeiluVision5的项目文件。

##### Keil uVision5

Keil uVision5是一款集成开发环境（IDE），用于编写、编译和调试嵌入式应用程序。它支持多种编程语言，包括C和汇编语言。Keil uVision5 **可以导入STM32CubeMX生成的项目文件**，并提供代码编辑、编译、调试和仿真功能。

##### 总结

综上所述，**STM32CubeMX**和Keil uVision5是两个可以协同工作的软件工具，前者用于生成初始化代码和配置文件，后者用于编写、编译和调试嵌入式应用程序，前者**生成的代码和配置文件可以轻松地导入到Keil uVision5中，以便进行进一步的开发和调试**。

- Q1：STM32CubeMX生成的初始化代码和配置文件为什么还需要Keil uVision5进行进一步的开发和调试
  - 因为STM32CubeMX生成的初始化代码和配置文件只是用于设置微控制器的硬件和基本操作，并不包含具体的业务逻辑。因此，需要使用Keil uVision5等集成开发环境（IDE）进行进一步的开发和调试。
- Q2：二者如何配合最后生成可执行文件
  - 在具体的开发过程中，开发人员需要在Keil uVision5中创建项目，编写应用程序代码，然后将代码与STM32CubeMX生成的初始化代码和配置文件进行链接和编译，最终生成可在目标板上运行的可执行文件。

#### [kiel5安装](https://blog.csdn.net/qq_59111928/article/details/132661448)

#### [STM32CubeMX安装](https://blog.csdn.net/weixin_68016945/article/details/130575044)

1. [取消自动更新检查](https://blog.csdn.net/Summer_taotao/article/details/130800817)

## 0.6 快速开始

1. 打开stm32 CubeMX，File --> New Project 

   ![image-20240517095242410](./legend/image-20240517095242410.png)

2. 选择芯片型号

   ![image-20240517101258645](./legend/image-20240517101258645.png)

3. a

4. 工程文件解析

   - 为了能让不同的芯片公司生产的Cortex-M4芯片能在软件上基本兼容，和芯片生产商共同提出了一套标准CMSIS标准(Cortex Microcontroller Software Interface Standard) ,翻译过来是“ARM Cortex™ 微控制器软件接口标准”。

   ![image-20240517104502625](./legend/image-20240517104502625.png)

5. 



# 1 GPIO

GPIO(General purpose input/output，通用型输入输出)

GPIO 是通过寄存器操作来实现电平信号输入输出的，对于输入，通过读取输入数据寄存器（IDR，Input Data Register）来确定引脚电位的高低；对于输出，通过写入输出数据寄存器（ODR,Output Data Register）来让这个引脚输出高电位或者低电位；对于其他特殊功能，则有另外的寄存器来控制它们。GPIO 的输出通常可以输出 0/1 信号，用来实现如 LED 灯、继电器、蜂鸣器等控制，而 GPIO 的输入可识别 0/1 信号，用来实现开关、按键等等动作或状态判定。

## 1.1 单引脚电路图

![](./legend/GPIO单引脚.png)

- 保护二极管
  - IO引脚上下两边两个二极管用于防止引脚外部过高、过低的电压输入。防止不正常电压引入芯片导致芯片烧毁
  - 当引脚电压高于VDD_FT时，上方的二极管导通
  - 当引脚电压低于VSS时，下方的二极管导通
- 上拉、下拉电阻
  - 控制引脚默认状态的电压，开启上拉的时候引脚默认电压为高电平，开启下拉的时候引脚默认电压为低电平
- TTL施密特触发器
  - 基本原理是当输入电压高于正向阈值电压，输出为高；当输入电压低于负向阈值电压，输出为低；
  - IO口信号经过触发器后，模拟信号转化为0和1的数字信号。
- P-MOS管和N-MOS管
  - 信号由P-MOS管和N-MOS管，依据两个MOS管的工作方式，使得GPIO具有“推挽输出”和“开漏输出”的模式
  - P-MOS管低电平导通，高电平关闭
  - 下方的N-MOS高电平导通，低电平关闭

## 1.2 8种工作模式

1. 浮空输入GPIO_Mode_IN_FLOATING

   - 浮空输入模式下，I/O端口的电平信号直接进入输入数据寄存器。
   - MCU直接读取I/O口电平，I/O的电平状态是不确定的，完全由外部输入决定，不开启上拉和下拉。
   - 如果在该引脚悬空（在无信号输入）的情况下，读取该端口的电平是不确定的
   - 特点：低功耗。
   - ![image-20240408201036462](./legend/image-20240408201036462.png)

2. 上拉输入GPIO_Mode_IPU（In Pull Up）

   - IO内部接上拉电阻
   - 如果IO口外部没有信号输入或者引脚悬空，IO口默认为高电平
   - 如果I/O口输入低电平，那么引脚就为低电平，MCU读取到的就是低电平。
   - 特点：钳位电平、增强驱动能力、抗干扰，可以用来检测外部信号
   - ![image-20240408201233261](./legend/image-20240408201233261.png)

3. 下拉输入GPIO_Mode_IPD（In Pull Down）

   - ![image-20240408201400560](./legend/image-20240408201400560.png)

4. 模拟输入GPIO_Mode_AIN（Analog Input）

   - 当GPIO引脚用于ADC采集电压的输入通道时，用作"模拟输入"功能，此时信号不经过施密特触发器，直接进入ADC模块，并且输入数据寄存器为空 ，CPU不能在输入数据寄存器上读到引脚状态
   - 当GPIO用于模拟功能时，引脚的上、下拉电阻是不起作用的，这个时候即使配置了上拉或下拉模式，也不会影响到模拟信号的输入
   - 

   - ![image-20240408201431290](./legend/image-20240408201431290.png)

5. 推挽输出GPIO_Mode_Out_PP（out push—pull）

   - 在推挽输出模式时，N-MOS管和P-MOS管都工作。
   - 如果我们控制IO输出为0，低电平，则P-MOS管关闭，N-MOS管导通，使输出低电平，I/O端口的电平就是低电平。若控制输出为1 高电平，则P-MOS管导通N-MOS管关闭，输出高电平，I/O端口的电平就是高电平。
   - 外部上拉和下拉的作用是控制在没有输出时IO口电平
   - 在这种模式下，施密特触发器是打开的，即输入可用，通过输入数据寄存器GPIOx_IDR可读取I/O的实际状态。此时I/O口的输入电平一定是输出的电平，一般应用在输出电平为0和3.3伏而且需要高速切换开关状态的场合。
   - ![image-20240408203716918](./legend/image-20240408203716918.png)

6. 开漏输出GPIO_Mode_Out_OD（out open drain）

   - 在开漏输出模式时，只有N-MOS管工作
   - 如果我们控制输出为0，低电平，则P-MOS管关闭，N-MOS管导通，使输出低电平，I/O端口的电平就是低电平。
   - 若控制输出为1时，高电平，则P-MOS管和N-MOS管都关闭，输出指令就不会起到作用。
   - 此时I/O端口的电平就不会由输出的高电平决定，而是由I/O端口外部的上拉或者下拉决定 如果没有上拉或者下拉 IO口就处于悬空状态。

7. 复用推挽输出GPIO_AF_PP（alternate function open push—pull）

   - GPIO 复用为其他外设(如 I2C)，输出数据寄存器GPIOx_ODR 无效
   - 输出的高低电平的来源于其它外设，施密特触发器打开，输入可用，通过输入数据寄存器可获取I/O 实际状态
   - 除了输出信号的来源改变，其他**与推挽输出功能相同**。应用于片内外设功能（I2C 的SCL,SDA）等。
   - ![image-20240408204433165](./legend/image-20240408204433165.png)

8. 复用开漏输出GPIO_AF_OD（alternate function open drain）

   - GPIO 复用为其他外设，输出数据寄存器GPIOx_ODR 无效；
   - 输出的高低电平的来源于其它外设，施密特触发器打开，输入可用，通过输入数据寄存器可获取I/O 实际状态
   - 除了输出信号的来源改变 其他**与开漏输出功能相同**。应用于片内外设功能（TX1,MOSI,MISO.SCK.SS）等。

9. 

## 1.3 GPIO接口

```c
// 初始化GPIOx引脚按照GPIO_Init指定的参数。
void HAL_GPIO_Init(GPIO_TypeDef* GPIOx, GPIO_InitTypeDef* GPIO_Init);
// 将GPIOx引脚重置为默认状态。
void HAL_GPIO_DeInit(GPIO_TypeDef* GPIOx, uint32_t GPIO_Pin);
// 读取指定GPIO端口的引脚状态。
GPIO_PinState HAL_GPIO_ReadPin(GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin);		// 返回引脚状态GPIO_PIN_SET 或 GPIO_PIN_RESET
// 设置或清除选定的GPIO端口引脚。
void HAL_GPIO_WritePin(GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin, GPIO_PinState PinState);
// 切换指定GPIO端口的引脚状态。
void HAL_GPIO_TogglePin(GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin);
//锁定GPIO引脚的配置，直到下一次复位。
HAL_StatusTypeDef HAL_GPIO_LockPin(GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin);
// 处理外部中断/事件请求。
void HAL_GPIO_EXTI_IRQHandler(uint16_t GPIO_Pin);
// 外部中断/事件线的回调函数。
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin);

// 在指定时间内阻塞。
void HAL_Delay(uint32_t Delay);

```

### LED闪烁

```c
GPIO_InitTypeDef GPIO_InitStruct = {0};
/*Configure GPIO pin : PB5 */
GPIO_InitStruct.Pin = GPIO_PIN_5;
GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
GPIO_InitStruct.Pull = GPIO_NOPULL;
GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

// led闪烁˸
// 查看原理图后，led灯低电平亮，高电平暗		
HAL_GPIO_WritePin(GPIOB,GPIO_PIN_5,GPIO_PIN_RESET);
HAL_Delay(1000);
HAL_GPIO_WritePin(GPIOB,GPIO_PIN_5,GPIO_PIN_SET);
HAL_Delay(1000);
```

读key调整led

```c
// 配置GPIO
GPIO_InitTypeDef GPIO_InitStruct = {0};

/* GPIO Ports Clock Enable */
__HAL_RCC_GPIOE_CLK_ENABLE();
__HAL_RCC_GPIOB_CLK_ENABLE();

/*Configure GPIO pin Output Level */
HAL_GPIO_WritePin(GPIOB, GPIO_PIN_5, GPIO_PIN_SET);

/*Configure GPIO pin : PE3 */
GPIO_InitStruct.Pin = GPIO_PIN_3;
GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
GPIO_InitStruct.Pull = GPIO_PULLUP;
HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

/*Configure GPIO pin : PB5 */
GPIO_InitStruct.Pin = GPIO_PIN_5;
GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
GPIO_InitStruct.Pull = GPIO_PULLUP;
GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);


  while (1)
  {
// led闪烁˸
// 查看原理图后，led灯低电平亮，高电平暗		
    if(HAL_GPIO_ReadPin(GPIOE,GPIO_PIN_3) == GPIO_PIN_RESET){
        HAL_Delay(100);
        if(HAL_GPIO_ReadPin(GPIOE,GPIO_PIN_3) == GPIO_PIN_RESET){
            HAL_GPIO_TogglePin(GPIOB,GPIO_PIN_5);
        }
    }
  }
```



# 2 系统时钟树

系统时钟树是微控制器内部运作的心脏，它决定了处理器和外设的工作频率。

这一架构包括多种时钟源，如**内部RC振荡器、外部晶振，以及更复杂的相位锁定环（PLL）系统**。

这些时钟源通过精心设计的时钟树分配给微控制器的不同部分，包括核心处理器、各种外设以及总线系统。

理解系统时钟的配置和分配是优化性能和功耗的关键，同时也对保证系统稳定运行至关重要。

对STM32的系统时钟树进行分析，可以帮助开发者更好地把握时钟管理的细节，确保硬件组件在正确的频率下高效运行。此外，时钟树的配置也直接影响到系统的响应速度和处理能力，是微控制器编程中不可忽视的重要组成部分。

## 2.1 时钟源

三种不同的时钟源可被用来驱动系统时钟(SYSCLK)：

- HSI振荡器时钟（高速系统内部时钟）
  - RC振荡器，频率为8MHZ
- HSE振荡器时钟（高速系统外部时钟）
  - 可接石英谐振器、陶瓷谐振器，或者接外部时钟源，其频率范围为4MHZ~16MHZ。
- PLL时钟（Phase-Locked Loop，锁相环时钟）
  - 是一种控制系统的结构
  - 在微控制器和嵌入式系统中，PLL被用作时钟系统的一部分，能够生成频率比输入时钟（通常是内部的低速时钟或者外部的晶体振荡器提供的时钟）更高的时钟信号（用作**倍频**）。
  - 倍频输出。其时钟输入源可选择为HSI/2、HSE或者HSE/2。倍频选择2~16倍，但是其输出频率最大不得超过72MHZ。

也包含2种二级时钟源

- LSI低速内部时钟。RC振荡器，频率为40KHZ。
- LSE低速外部时钟。接频率为32.768KHZ的石英晶体。

# 3 中断NVIC

NVIC（Nest Vector Interrupt Controller），嵌套向量中断控制器，作用是管理中断嵌套，核心任务是管理中断优先级。

特点：

1. 68个可屏蔽中断通道(不包含16个Cortex-M3的中断线)
2. 16个可编程的优先等级(使用了4位中断优先级)
3. 低延迟的异常和中断处理
4. 电源管理控制
5. 系统控制寄存器的实现

## 3.1 优先级

NVIC给每个中断赋予**抢占优先级和响应优先级**。

抢占优先级决定了中断是否能够打断其他正在执行的中断。具有高抢占优先级的中断可以打断低抢占优先级的中断，形成中断嵌套。

响应优先级则是在抢占优先级相同的情况下用来决定哪个中断先被处理的。

关系如下：

1. 拥有较高抢占优先级的中断可以打断抢占优先级较低的中断
2. 若两个**抢占优先级相同**的中断同时挂起，则优先执行**响应优先级较高**的中断
3. 若两个**响应优先级相同**的中断同时挂起，则优先执行位于**中断向量表中位置较高**的中断
4. **响应优先级**不会造成中断嵌套，也就是说中断嵌套是由**抢占优先级决定的**

## 3.2 [外部中断](https://blog.csdn.net/qq_44016222/article/details/123539693)

### 外部中断线

- STM32的每个IO都可以作为外部中断输入。	
- STM32的中断控制器支持19个外部中断/事件请求（19个外部中断线）：
  - 线0~15：对应外部IO口的输入中断。
  - 线16：连接到PVD输出。
  - 线17：连接到RTC闹钟事件。
  - 线18：连接到USB唤醒事件。
- 每个外部中断线可以独立的配置触发方式（上升沿，下降沿或者双边沿触发），触发/屏蔽，专用的状态位。

### 外部中断线与IO引脚对应关系

如下图：

![img](legend/外部中断.png)



### 中断向量与服务函数

是不是16个中断线就可以分配16个中断服务函数呢？IO口外部中断在中断向量表中只分配了7个中断向量，也就是只能使用7个中断服务函数

![img](./legend/中短线与服务函数.png)



从表中可以看出，外部中断线5~9分配一个中断向量，共用一个服务函数外部中断线10~15分配一个中断向量，共用一个中断服务函数。



![image-20240521145030191](./legend/image-20240521145030191.png)

每个中断源都需要被指定这两种优先级，Cortex-M3核定义了8个bit用于设置中断源的优先级。

但是Cortex-M3允许具有较少中断源时使用较少的寄存器位指定中断源的优先级，因此STM32中断优先级的寄存器位只用到AIRCR高四位。

```c
// 配置中断优先级，数字越低，中断优先级越高
HAL_NVIC_SetPriority(IRQn_Type IRQn, uint32_t PreemptPriority, uint32_t SubPriority);
// 参数：
typedef enum { 
    	/****** Cortex-M3 Processor Exceptions Numbers ***************************************************/ 
    	NonMaskableInt_IRQn = -14, /*!< 2 不可屏蔽中断 */
    	HardFault_IRQn = -13, /*!< 3 Cortex-M3 硬故障中断 */
        MemoryManagement_IRQn = -12, /*!< 4 Cortex-M3 内存管理中断 */ 
        BusFault_IRQn = -11, /*!< 5 Cortex-M3 总线故障中断 */
        UsageFault_IRQn = -10, /*!< 6 Cortex-M3 使用故障中断 */ 
        SVCall_IRQn = -5, /*!< 11 Cortex-M3 SV 调用中断 */ 
        DebugMonitor_IRQn = -4, /*!< 12 Cortex-M3 调试监视器中断 */
        PendSV_IRQn = -2, /*!< 14 Cortex-M3 Pend SV 中断 */ 
        SysTick_IRQn = -1, /*!< 15 Cortex-M3 系统滴答定时器中断 */ 
        
        /****** STM32 specific Interrupt Numbers *********************************************************/ 
        WWDG_IRQn = 0, /*!< 窗口看门狗中断 */
        PVD_IRQn = 1, /*!< PVD 通过 EXTI 线检测中断 */
        TAMPER_IRQn = 2, /*!< 篡改中断 */
        RTC_IRQn = 3, /*!< RTC 全局中断 */
        FLASH_IRQn = 4, /*!< FLASH 全局中断 */
        RCC_IRQn = 5, /*!< RCC 全局中断 */
        EXTI0_IRQn = 6, /*!< EXTI 线0 中断 */
        EXTI1_IRQn = 7, /*!< EXTI 线1 中断 */
        EXTI2_IRQn = 8, /*!< EXTI 线2 中断 */
        EXTI3_IRQn = 9, /*!< EXTI 线3 中断 */
        EXTI4_IRQn = 10, /*!< EXTI 线4 中断 */
        DMA1_Channel1_IRQn = 11, /*!< DMA1 通道1全局中断 */
        DMA1_Channel2_IRQn = 12, /*!< DMA1 通道2全局中断 */
        DMA1_Channel3_IRQn = 13, /*!< DMA1 通道3全局中断 */
        DMA1_Channel4_IRQn = 14, /*!< DMA1 通道4全局中断 */
        DMA1_Channel5_IRQn = 15, /*!< DMA1 通道5全局中断 */
        DMA1_Channel6_IRQn = 16, /*!< DMA1 通道6全局中断 */
        DMA1_Channel7_IRQn = 17, /*!< DMA1 通道7全局中断 */
        ADC1_2_IRQn = 18, /*!< ADC1 和 ADC2全局中断 */
        USB_HP_CAN1_TX_IRQn = 19, /*!< USB 设备高优先级或CAN1 TX 中断 */
        USB_LP_CAN1_RX0_IRQn = 20, /*!< USB 设备低优先级或CAN1 RX0 中断 */
        CAN1_RX1_IRQn = 21, /*!< CAN1 RX1 中断 */
        CAN1_SCE_IRQn = 22, /*!< CAN1 SCE 中断 */
        EXTI9_5_IRQn = 23, /*!< 外部线[9:5]中断 */
        TIM1_BRK_IRQn = 24, /*!< TIM1 Break 中断 */
        TIM1_UP_IRQn = 25, /*!< TIM1 Update 中断 */
        TIM1_TRG_COM_IRQn = 26, /*!< TIM1 Trigger and Commutation 中断 */
        TIM1_CC_IRQn = 27, /*!< TIM1 Capture Compare 中断 */
        TIM2_IRQn = 28, /*!< TIM2全局中断 */
        TIM3_IRQn = 29, /*!< TIM3全局中断 */
        TIM4_IRQn = 30, /*!< TIM4全局中断 */
        I2C1_EV_IRQn = 31, /*!< I2C1 事件中断 */
        I2C1_ER_IRQn = 32, /*!< I2C1 错误中断 */
        I2C2_EV_IRQn = 33, /*!< I2C2 事件中断 */
        I2C2_ER_IRQn = 34, /*!< I2C2 错误中断 */
        SPI1_IRQn = 35, /*!< SPI1全局中断 */
        SPI2_IRQn = 36, /*!< SPI2全局中断 */
        USART1_IRQn = 37, /*!< USART1全局中断 */
        USART2_IRQn = 38, /*!< USART2全局中断 */
        USART3_IRQn = 39, /*!< USART3全局中断 */
        EXTI15_10_IRQn = 40, /*!< 外部线[15:10]中断 */
        RTC_Alarm_IRQn = 41, /*!< RTC闹钟通过 EXTI 线中断 */
        USBWakeUp_IRQn = 42, /*!< USB 设备从待机状态通过 EXTI 线中断唤醒 */
        TIM8_BRK_IRQn = 43, /*!< TIM8 Break 中断 */
        TIM8_UP_IRQn = 44, /*!< TIM8 Update 中断 */
        TIM8_TRG_COM_IRQn = 45, /*!< TIM8 Trigger and Commutation 中断 */
        TIM8_CC_IRQn = 46, /*!< TIM8 Capture Compare 中断 */ 
        ADC3_IRQn = 47, /*!< ADC3全局中断 */
        FSMC_IRQn = 48, /*!< FSMC全局中断 */
        SDIO_IRQn = 49, /*!< SDIO全局中断 */
        TIM5_IRQn = 50, /*!< TIM5全局中断 */
        SPI3_IRQn = 51, /*!< SPI3全局中断 */
        UART4_IRQn = 52, /*!< UART4全局中断 */
        UART5_IRQn = 53, /*!< UART5全局中断 */
        TIM6_IRQn = 54, /*!< TIM6全局中断 */
        TIM7_IRQn = 55, /*!< TIM7全局中断 */
        DMA2_Channel1_IRQn = 56, /*!< DMA2 通道1全局中断 */
        DMA2_Channel2_IRQn = 57, /*!< DMA2 通道2全局中断 */
        DMA2_Channel3_IRQn = 58, /*!< DMA2 通道3全局中断 */
        DMA2_Channel4_5_IRQn = 59 /*!< DMA2 通道4和通道5全局中断 */
} IRQn_Type;
uint32_t PreemptPriority; // 抢占优先级，根据优先级分组确定。 
uint32_t SubPriority);  // 响应优先级，根据优先级分组确定。

// 在NVIC中启用特定于设备的中断
void HAL_NVIC_EnableIRQ(IRQn_Type IRQn);
// 在NVIC中禁用特定于设备的中断
void HAL_NVIC_DisableIRQ(IRQn_Type IRQn);
```

### 开关灯

key1按下开灯，key0按下关灯

![](./legend/中断开关灯.png)

```c
// gpio.c
void MX_GPIO_Init(void)
{

  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_5, GPIO_PIN_RESET);

  /*Configure GPIO pins : PE3 PE4 */
  GPIO_InitStruct.Pin = GPIO_PIN_3|GPIO_PIN_4;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /*Configure GPIO pin : PE5 */
  GPIO_InitStruct.Pin = GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI3_IRQn);

  HAL_NVIC_SetPriority(EXTI4_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI4_IRQn);

}

// stm32f1xx_it.c
// @brief This function handles EXTI line3 interrupt.
void EXTI3_IRQHandler(void)
{
  /* USER CODE BEGIN EXTI3_IRQn 0 */

  /* USER CODE END EXTI3_IRQn 0 */
  HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_3);		// go to definition Of HAL_GPIO_EXTI_IRQHandler， 可以在stm32f1xx_hal_gpio.c找到这个函数
  /* USER CODE BEGIN EXTI3_IRQn 1 */

  /* USER CODE END EXTI3_IRQn 1 */
}
// @brief This function handles EXTI line4 interrupt.
void EXTI4_IRQHandler(void)
{
  /* USER CODE BEGIN EXTI4_IRQn 0 */

  /* USER CODE END EXTI4_IRQn 0 */
  HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_4);
  /* USER CODE BEGIN EXTI4_IRQn 1 */

  /* USER CODE END EXTI4_IRQn 1 */
}

// 可以从stm32f1xx_hal_gpio.c中的HAL_GPIO_EXTI_IRQHandler，看到HAL_GPIO_EXTI_Callback，并且在HAL_GPIO_EXTI_IRQHandler函数的下面看到HAL_GPIO_EXTI_Callback的定义，我们需要对它进行重写，写在main.c中

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
	if(GPIO_Pin == GPIO_PIN_3){	// key1
        // 开灯
		HAL_GPIO_WritePin(GPIOE,GPIO_PIN_5,GPIO_PIN_RESET);
	}else if(GPIO_Pin == GPIO_PIN_4){ // key0
        // 关灯
		HAL_GPIO_WritePin(GPIOE,GPIO_PIN_5,GPIO_PIN_SET);
	}
}
```

# 4 定时器

STM32的定时器是单片机内部的硬件外设，它们嵌入在STM32芯片的硅片中，不是外部组件。这里提到的“外设”一词是指这些定时器是相对于微控制器的核心CPU而言的，它们是微控制器的一部分，提供了额外的功能，但它们并不是物理上独立于STM32芯片的外部设备。

STM32微控制器的设计集成了多种外设，包括但不限于定时器、ADC（模数转换器）、DAC（数模转换器）、UART（通用异步收发传输器）、SPI（串行外设接口）、I2C（集成电路总线）、USB（通用串行总线）等。这些外设都集成在STM32芯片内部，可以在不需要外部组件的情况下实现多种功能，从而简化了电路设计和降低了成本。

STM32总共有8个定时器，分别是2个高级定时器（TIM1、TIM8），4个通用定时器（TIM2、TIM3、TIM4、TIM5）和2个基本定时器（TIM5、TIM6）。

每个定时器都是一个可编程预分频器驱动的16位自动装载计数器构成 每个定时器都是完全独立的，没有互相共享任何资源。

![image-20240521204955925](./legend/image-20240521204955925.png)



# 其它

1. 代码跳转需要先编译后才能跳转
2. [cubemx在Pinout view中选错了引脚，不知道怎么取消](https://blog.csdn.net/qq_52932171/article/details/132240143)
   - 左键要取消的引脚，再点击一次之前设置的模式就可取消
   - ![img](./legend/8878a0509b0949f3bccfb53bc22d8892.png)
3. [internal commend error](https://blog.csdn.net/qq_60341895/article/details/127629430)
4. [快捷键](https://blog.csdn.net/qq_44250317/article/details/125635828)
