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

# vscode debug

由于想在node.js代码中，使用alias `@` 用于代表`/src`目录下，方便解决引用的麻烦。

这时我们可以使用，webpack或者babel构建工具，对自己的js源代码进行预处理构建。这些构建工具通常是对源代码进行文本处理，或者js语法的版本转化

本次我们在electron中，使用babel对源代码进行处理。以达到对alias `@` 符的引用处理。



构建后的代码，格式比较无序，而且使用的js代码的版本也发生了转换，甚至换行符也消失了，所以要想调试代码，必须将构建后的代码映射到源代码上去。

在vscode中，要调试构建后的代码，除了babel等构建工具必须建立source-map外，也需要在vscode进行配置。

在项目的根目录下，建立一个.vscode文件夹。



首先在在launch.json中配置：

launch.json配置项的功能解释如下（下面只做解释）：

```js
{
  "version": "0.2.0",
  "configurations": [
    {
        // 用于在调试配置列表中显示。debug的名字，在debug界面可以看到 
      "name": "Launch Program",                
        // 调试器的类型，例如 node、python、cppdbg 等。
      "type": "node",
        //
      "request": "launch",
        // 要调试的程序的入口文件路径。通产指构建后的入口文件，而不是源文件
      "program": "${workspaceFolder}/dist/app.js",
        // 传递给程序的命令行参数。
      "args": ["arg1", "arg2"],
        // 设置程序的工作目录。
      "cwd": "${workspaceFolder}",
        // 设置环境变量。
      "env": {
        "NODE_ENV": "development"
      },
        // 是否启用源映射支持，使调试器能够将编译后的代码映射回源代码。
      "sourceMaps": true,
        // 指定构建后的 JavaScript 文件的位置，这里是 dist 目录下的所有 .js 文件。调试器会使用这些文件中的源映射信息来映射回源代码。
      "outFiles": ["${workspaceFolder}/dist/**/*.js"],
        // 通过正确配置 outFiles，你可以确保调试器能够正确加载和使用源映射文件，从而在调试时看到源代码而不是编译后的代码。
        
        //在启动调试会话之前运行的任务，通常用于构建项目。
      "preLaunchTask": "build",
      "postDebugTask": "cleanup",
        // 指定要使用的运行时可执行文件的路径
      "runtimeExecutable": "${workspaceFolder}/node_modules/.bin/electron",
        // 传递给运行时可执行文件的参数。
      "runtimeArgs": ["--inspect"],
      "console": "integratedTerminal",
      "stopOnEntry": false,
      "internalConsoleOptions": "openOnSessionStart"
    }
  ]
}
```



实际的在项目中我们如下配置launch.json：

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "type": "node",
            "request": "launch",
            "name": "babel",
            "runtimeExecutable": "${workspaceFolder}/node_modules/.bin/electron",
            "program": "${workspaceFolder}/dist/main/index.js",
            "sourceMaps": true,
            "env": {
                "BABEL_ENV": "debug",
                "NODE_ENV": "development"
              },
            "outFiles": ["${workspaceFolder}/dist/**/*.js"],
            "preLaunchTask": "npm: bbuild"
        }
    ]
}
```

然后在tasks.json中

```js
{
    "version": "2.0.0",
    "tasks": [
      {
        "label": "npm: bbuild",
        "type": "npm",
        "script": "bbuild",
        "group": "build",
        "problemMatcher": [],
        "detail": "This task runs the bbuild script defined in package.json."
      }
    ]
  }
```



package.json内容如下

```json
{
  "name": "gai",
  "version": "1.0.0",
  "description": "gai desktop",
  "main": "dist/main/index.js",
  "scripts": {
    "bbuild": "babel src --out-dir dist --source-maps"
},
"author": "q",
  "license": "ISC",
  "devDependencies": {
    "@babel/cli": "^7.25.7",
    "@babel/core": "^7.25.8",
    "@babel/preset-env": "^7.25.8",
    "babel-plugin-module-resolver": "^5.0.2"
},
  "dependencies": {
    "@babel/node": "^7.25.7",
    "electron-log": "^5.2.0",
    "vite": "^5.4.9",
    "vite-plugin-require": "^1.2.14"
  }
```



