# ROS

**ROS全称Robot Operating System(机器人操作系统)**

ROS设计者将ROS表述为“ROS = Plumbing + Tools + Capabilities + Ecosystem”，即ROS是通讯机制、工具软件包、机器人高层技能以及机器人生态系统的集合体。

ROS主要设计了如下目标：

- **代码复用:**ROS的目标不是成为具有最多功能的框架，ROS的主要目标是支持机器人技术研发中的代码*重用*。
- **分布式:**ROS是进程（也称为*Nodes*）的分布式框架,ROS中的进程可分布于不同主机，不同主机协同工作，从而分散计算压力
- **松耦合:**ROS中功能模块封装于独立的功能包或元功能包，便于分享，功能包内的模块以节点为单位运行，以ROS标准的IO作为接口，开发者不需要关注模块内部实现，只要了解接口规则就能实现复用,实现了模块间点对点的松耦合连接
- **精简：**ROS被设计为尽可能精简，以便为ROS编写的代码可以与其他机器人软件框架一起使用。ROS易于与其他机器人软件框架集成：ROS已与OpenRAVE，Orocos和Player集成。
- **语言独立性：**包括Java，C++，Python等。为了支持更多应用开发和移植，ROS设计为一种语言弱相关的框架结构，使用简洁，中立的定义语言描述模块间的消息接口，在编译中再产生所使用语言的目标文件，为消息交互提供支持，同时允许消息接口的嵌套使用
- **易于测试：**ROS具有称为[rostest](http://wiki.ros.org/rostest)的内置单元/集成测试框架，可轻松安装和拆卸测试工具。
- **大型应用：**ROS适用于大型运行时系统和大型开发流程。
- **丰富的组件化工具包：**ROS可采用组件化方式集成一些工具和软件到系统中并作为一个组件直接使用，如RVIZ（3D可视化工具），开发者根据ROS定义的接口在其中显示机器人模型等，组件还包括仿真环境和消息查看工具等
- **免费且开源：**开发者众多，功能包多

ROS 目前已经发布了ROS1 的终极版本: noetic，并建议后期过渡至 ROS2 版本。

[ros1的发行版和维护时间](http://wiki.ros.org/Distributions)

[ROS2的发行版和维护时间](https://docs.ros.org/en/foxy/Releases.html)

## 0 绪论

## 0.1 ROS1和ROS2

[区别](https://blog.51cto.com/u_16099273/8716312)

![ROS和ROS docker版的区别 ros和ros2区别_linux](./legend/resize,m_fixed,w_1184.webp)

中间层：通信机制，DDS（Data Distribution Sevice）



## 0.1 安装ROS

