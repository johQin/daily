# ADB

**ADB 属于 Android SDK，Android SDK 运行需要依赖 JDK**

Android Debug Bridge，安卓设备官方调试工具，**不用装额外软件，电脑就能控制 PDA**

ADB 能控制系统、读系统状态、模拟软件层面操作；

## 常用命令

```bash
# 查看adb的版本
adb version
# 连接设备
adb connect 127.0.0.1:62001
# 查看有哪些设备，设备的状态：device 正常，unknown 没有连接设备，offline 设备连接异常
adb devices
# 查看设备状态
adb get-state

# 查看日志，是实时日志，Ctrl + C 可退出查看
adb logcat -b buffer_area
# 缓冲区默认是：main，system
# 可查看手机四个缓冲区日志，包括：radio，system，main，event
# radio，通信日志，eg:通过短信
# system，系统组件日志
# main，手机应用软件日志，应用层的日志输出
# event，手机按键输出等事件日志等
# -c 清理日志


# 输出日志包括logcat 日志，同时也包含功耗，cpu等日志信息
adb bugreport
# 安装app应用
adb install d:/xxx.apk
# 卸载app应用
adb uninstall -k <packagename>

# 进入root模式
adb root
# 进入shell命令状态，android 基于linux内核，因此在shell状态下可以使用linux命令
adb shell
# 后面所有shell开头的，都执行的是手机内部的命令

# 包名获取
# 所有包名
adb shell pm list packages
# 查看当前活动包的信息
adb shell dumpsys activity
# 查看当前正在运行的包名‘
adb shell dumpsys window | findstr mCurrentFocus

# 获取cpu信息
adb shell cat /proc/cpuinfo
# 查看手机分辨率
adb shell wm size
# 查看电池信息
adb shell dumpsys battery
# 查看应用的耗电情况
adb shell dumpsys batterystats cn.monpon.film 
# 查看系统版本号
adb shell getprop or.build.version.release
# 
adb kill-server
adb start-server
# 手机电脑相互copy文件
adb push d:\pushlog.txt /dev/log
adb pull /dev/log/error.log d:\

# 截屏
adb shell screencap -p /sdcard/1.png

adb shell
su
# 查看连接过的wifi信息
cat /data/misc/wifi/*.conf
# mac地址
cat /sys/class/net/wlan0/address
# 
cat /procf/meminfo

# 重启手机
adb reboot


```

