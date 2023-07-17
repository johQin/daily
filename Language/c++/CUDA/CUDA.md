# CUDA

# 1 环境搭建

```bash
# 查看电脑的显卡
lspci | grep -i vga



```

## 1.1 安装问题纪实

[安装教程](https://blog.csdn.net/weixin_39928010/article/details/131142603)

```bash
# 1. 安装驱动

# Error : your appear to running an x server；please exit x before installing .for further details
# 解决方案： https://blog.csdn.net/qq_32415217/article/details/123185645
sudo chmod +x NVIDIA-Linux-x86_64-535.54.03.run
sudo ./NVIDIA-Linux-x86_64-535.54.03.run -no-x-check

# ERROR: The Nouveau kernel driver is currently in use by your system. This driver is incompatible with the NVIDIA driver……

```

