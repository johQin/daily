# Vite

vite作为前端框架构建工具，和webpack作用类似，不过vite很多配置都是默认的，所以不需要写太多配置的内容，即可构建一个前端框架。

# 1 vite构建vue项目

```bash
# 按照引导逐步填写项目信息
npm init vite
# 下面两种都可以使用vite命令构建项目
# npx create vite 
# yarn create vite
```





# 2 vite 配置

## 2.1 模式

Vite 默认是`development`，读取项目根路径下`.env.development`配置文件

vite build 默认是`production`，读取项目根路径下`.env.production`配置文件

`vite build --mode myenv`，自定义模式，读取项目根路径下`.env.myenv`配置文件

## 2.2 配置文件

vite默认读取项目根路径下`vite.config.js`文件。

可以指定不同配置文件：

- `vite --config vite.config.dev.js`
- `vite build --config vite.config.prod.js`