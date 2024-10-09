# Electron

# 0 环境搭建

```bash
# 如果以普通的方式安装，将会报：postinstall: node install.js的错
# npm install electron
# 这就是包周期脚本出问题了

# 首先以这种方式安装，然后在后面使用electron命令，就会报：Error: Electron failed to install correctly, please delete node_modules/electron and try installing again
npm install electron --ignore-scripts
# npm --ignore-scripts 参数用于在安装 npm 包时忽略执行 package.json 中定义的生命周期脚本（lifecycle scripts）。

#要解决上面的问题，可以参考：https://blog.csdn.net/weixin_44582045/article/details/128798694文章的解决方法的方法1
# 先下载electron已编译好的包：https://registry.npmmirror.com/binary.html?path=electron/32.1.2/
# 也可以在github electron仓库中下载
# 其中的：electron-v32.1.2-win32-x64.zip
# 然后在node_modules\electron\下创建dist文件夹。
# 将下载的压缩包解压进刚刚创建的dist。
# 在node_modules\electron\中创建path.txt，内容为electron.exe（对应自己的平台，不同平台不一样）。
```



# 包

## cross-env

cross-env 的工作原理是：它在执行命令之前设置环境变量，并在命令执行完毕后清除这些环境变量。这样可以确保环境变量只在需要的命令执行期间存在，而不会影响其他命令或全局环境。

```json
{
  "scripts": {
    "build": "cross-env NODE_ENV=production webpack --config build/webpack.config.js"
  }
}
```

## electron-builder

[electron-builder打包过程中报错——网络下载篇](https://blog.csdn.net/qq_32682301/article/details/105234408)
