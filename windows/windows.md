# windows

# 1 文件系统

```bash
#1.切换文件夹
cd .
cd ..
cd \ #根目录
E:#盘符切换
#2.查看目录
dir
#3.创建/删除/复制文件夹
md 文件夹名/相对绝对路径
rd 文件夹名
xcopy 路径1/文件夹1 路径2/文件夹2 /s /e #/s 连同子文件一起复制
#4.创建/删除/复制文件
type nul>test.txt #创建空文件
del 文件名
copy 路径1/文件名1 路径2/文件名2


```

# 2 [远程连接](<https://jingyan.baidu.com/article/e8cdb32bfc4f3137052bad03.html>)

被控电脑

我的电脑->属性->远程设置->远程桌面->仅允许运行使用网络级别身份验证

主控电脑

win+r->mstsc->输入被控电脑ip->输入用户名和电脑密码即可连接

# 3 快捷键

1. 桌面双显的快捷设置：【 Win + P 】
2. 显示桌面：【Win + D】
3. 软件应用分屏：【Win + ← →】
4. 命令行：【 Win + R ，输入CMD，Enter 】
5. 选中当前单词：【Ctrl + Shift + ← →】