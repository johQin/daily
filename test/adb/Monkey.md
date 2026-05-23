# Monkey

Monkey 本身是android 系统中的一个命令行工具。

## 初识

测试场景：

1. app的**压力测试，稳定性测试**
2. app的功耗，内存，cpu

特点：

1. 没法指定具体业务（无法做功能测试），发送**伪随机**事件（触屏，划屏，按键等）
2. 主要通过参数来做一些设定

monkey的位置在：

- 存在于Android系统中，/system/framework/monkey.jar（java工具），通过/system/bin/monkey的命令去调用这个jar包

安装 android “夜神模拟器”，打开模拟器

电脑连接模拟器

```bash
adb devices
# 连接andorid 模拟器，夜神的端口时62001
adb connect 127.0.0.1:62001
# 查看是否连接成功
adb devices
List of devices attached
127.0.0.1:62001 device
# 进入android里面的命令行
adb shell


```

## 环境

**ADB 属于 Android SDK，Android SDK 运行需要依赖 JDK**

先安装jdk，然后安装Android SDK



```bash
adb shell monkey
```

## Android Monkey 参数

Monkey 命令格式：`adb shell monkey [参数] <事件次数count>`

参数主要分为 **基础类、事件类、约束类、调试类、日志类、安全 / 权限类、其他辅助类** 7 大类，

**基本参数**：

- -P：应用的包名
- -v：显示日志，控制日志的详细程度，`-v -v`，`-v -v -v`最详细
- -S：seed 种子数，随机的种子数，不同的种子数，对应不同的随机事件。相同的种子数，可以对应相同的随机序列，在某些场景下，可能需要相同的种子数才能复现问题。
- --throttle：事件间隔时间，单位毫秒

```bash
# 这可以查看android手机中所有的包名
adb shell pm list packages
package:com.taobao.taobao
# 查看第三方的报名，像taobao等等
adb shell pm list packages -3

# 查看当前活动窗的包名
adb shell dumpsys window | findstr "mCurrentFocus"

# 查看这个cn.mopon.film 事件100次的日志
adb shell monkey -p cn.mopon.film -v -v 100

adb shell monkey -p cn.mopon.film -s 102 -v -v --throttle 300 100 > d:\app\monkey102.log
```

**事件参数**：

- 可以通过事件参数指定你要的事件，之间的比例，总比例不超过100

```bash
adb shell monkey -p cn.mopon.film -s 102 -v -v --throttle 300 --pct-motion 40 --pct-rotation 30 --ignore-crashes 100 > d:\app\monkey102.log
```

## 看日志

通过 **>** 将日志输出到文件后，查看文件的最后一行是否有：**// Monkey finished**，这表明当前这条命令是否正常退出。

出错原因：

1. 闪退，崩溃，重启，关机（adb logcat）
2. **// exception**：查看日志中是否有exception
3. **// crash**：查看日志中是否有Crash
4. **// ANR**：Application Not Response 无响应