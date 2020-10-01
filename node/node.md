# 1.前言
## 1.1what 

Node.js是一个基于Chrome V8引擎的JavaScript运行时。

### 特点
- chrome v8 runtime
- 事件驱动
- 非阻塞的I/O
  
    同时大量的数据处理性能好——高并发特别好。

    网络请求 数据库处理 文件的读写都是阻塞的同步的。
## 1.2why
- 写api
- 了解前后端的交互流程
- 全栈
# 2.js
    出于安全性考虑，进行前后端分离，不是说js语言上无法实现。
## 2.1 前端js
- 基本语法
- Bom
- Dom
- ajax
## 2.2 服务器端js
- 基本语法
- 操作数据库
- 操作本地文件

限制语言能力的不是语言本身，而是运行环境（平台）

# 3. node.js

## 3.1 nvm
    node-version-manage node版本管理工具
- nvm ls 查看已安装的node 所有版本  
- nvm use v10.8.0 切换到对应版本
## 3.2 REPL
    node 运行环境
```bash
    >node       //cmd 内输入，可以直接运行js代码
    >1+1
    >2
```
```cmd
    node filePath/xxx.js    //运行js文件
```
## 3.3 模块化
- 内置模块（node提供，直接用）
- 第三方模块
- 自定义模块
    - 创建一个模块（一个js就是一个模块）
    - 导出一个模块（module.exports= name)
    - 引用一个模块并且调用(const fs=require('filepath');)
    

# 4 内置模块

## 4.1 [文件系统 fs](http://nodejs.cn/api/fs.html#fs_file_system)

`fs` 模块提供了用于与文件系统进行交互（以类似于标准 POSIX 函数的方式）的 API。

```js
const fs=require('fs');
//创建文件，覆盖写入
fs.writeFile()
//追加写入
fs.appendFile()
//读取文件
fs.readFile()
//删除文件
fs.unlink()
//文件信息
//fs.Stats对象提供了关于文件的信息,他可以从 fs.stat()、fs.lstat()、fs.fstat()、以及它们的同步方法返回的对象都是此类型。

```

## 4.2 URL

`url` 模块用于处理与解析 URL。

统一资源定位符

```JS
const url = require('url');
url.parse()

```

# 命令

1. 安装包：【npm install package -g/-D】-g，全局安装，-D项目安装
2. 查询包列表：【npm ls --depth 0】
3. 调用项目内部安装的模块：npx package