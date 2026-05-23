# Appium

Appium是一个开源项目和相关软件生态系统，旨在促进多种应用平台的用户界面自动化，包括移动端（iOS、Android、Tizen）、浏览器端（Chrome、Firefox、Safari）、桌面端（macOS、Windows）、电视端（Roku、tvOS、Android TV、三星）等！

# 0 初识

## 0.1 核心概念

- Appium Core：定义核心API
- Drivers：实现连接在指定的平台
- Clients：各种编程语言的API
- Plugins：改变或者扩展核心功能

## 0.2 架构原理

```
┌──────────────┐     HTTP      ┌──────────────┐     Native      ┌──────────────┐
│              │ ────────────> │              │ ──────────────> │              │
│  Test Script │   Response    │   Appium     │    Command      │  UiAutomator2│
│  (Python)    │ <──────────── │   Server     │ <────────────── │  / XCUITest  │
│              │               │  (Node.js)   │     Result      │              │
└──────────────┘               └──────────────┘                 └──────────────┘
                                       │
                                       │ 选择驱动
                                       ▼
                              ┌──────────────────┐
                              │  Platform Driver │
                              │  - UiAutomator2  │
                              │  - XCUITest      │
                              │  - Espresso      │
                              └──────────────────┘

```



1. 测试脚本通过 Appium 客户端库发送 HTTP 请求到 Appium Server
2. Appium Server 根据 Desired Capabilities 选择对应的平台驱动
3. 平台驱动将命令翻译为平台原生 API 调用
4. 设备执行操作并将结果逐层返回

## 0.3 环境准备

安装jdk，android sdk

安装appium

```bash
# 安装 Node.js（Appium 依赖）
brew install node

# 安装 Appium Server
npm install -g appium

# 安装 Appium Doctor（环境检查工具）
npm install -g appium-doctor

# 安装驱动
appium driver install uiautomator2    # Android 驱动
appium driver install xcuitest        # iOS 驱动

# 检查环境
appium-doctor --android
appium-doctor --ios

```

