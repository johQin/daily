# [FRP](https://blog.csdn.net/weixin_40483369/article/details/121210349)

frp 是一个专注于[内网穿透](https://so.csdn.net/so/search?q=内网穿透&spm=1001.2101.3001.7020)的高性能的反向代理应用，支持 TCP、UDP、HTTP、HTTPS 等多种协议。可以将内网服务以安全、便捷的方式通过具有公网 IP 节点的中转暴露到公网。

[参考](https://huaweicloud.csdn.net/63a567a2b878a5454594675f.html)

[frp源码](https://gitcode.net/mirrors/fatedier/frp)

# http穿透示例

这个示例通过简单配置 HTTP 类型的代理让用户访问到内网的 Web 服务。

HTTP 类型的代理相比于 TCP 类型，不仅在服务端只需要监听一个额外的端口 vhost_http_port 用于接收 HTTP 请求，还额外提供了基于 HTTP 协议的诸多功能。

## 需要两台服务器

云上服务器 1.15.180.135
本地服务器 192.168.1.48

FRP下载地址：https://github.com/fatedier/frp/releases

服务器是ubuntu，本地是windows，下两个不同的包。

## 云服务器frp配置

修改 frps.ini 文件，设置监听 HTTP 请求端口为 8081：

```ini
[common]
bind_port = 7000		
vhost_http_port = 8081
```

运行：

```bash
 ./frps -c ./frps.ini
```

## 本地服务器配置

修改 frpc.ini 文件，假设 frps 所在的服务器的 IP 为 1.15.180.135，local_port 为本地机器上 Web 服务监听的端口, 绑定自定义域名为 custom_domains。

 local_port 因为我本地服务的端口是8082

```ini
[common]
# 云服务器ip地址
server_addr = 1.15.180.135
server_port = 7000

[web]
# 提供的服务协议类型
type = http
# 本地服务器端口
local_port = 8082
# 本地服务器ip
local_ip = 192.168.0.104
# 云服务器ip地址
custom_domains = 1.15.180.135
```

运行

```bash
./frpc.exe -c ./frpc.ini
```

### 外网地址请求

```bash
# 浏览器访问
http://1.15.180.135:8081/

# 这样你就可以通过云服务器的1.15.180.135:8081，访问内网的192.168.0.104:8082
```

