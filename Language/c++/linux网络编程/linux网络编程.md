# linux网络编程

1. 计算机网络概念，七层和四层模型，协议族（TCP/IP），mac地址，IP地址，端口port，子网掩码，
2. 数据包的组包和拆包流程，TCP/UDP特点
3. UDP编程，编程准备（字节序，端口，ip，大小端转换的函数），UDPAPI，发送/接收数据
4. UDP-TFTP编程，udp广播，多播
5. TCP编程，客户端和服务器编写流程，三次握手，四次挥手
6. TCP高并发服务器，多进程，多线程服务器，select，poll，epoll实现的服务器，epoll+线程池
7. 网络通信过程
8. 原始套接字，自己组底层的数据包，收一帧完整的数据包
9. web服务器

# 1. 计算机网络概念

## 1.1 TCP/IP协议族



### 1.3.1 分层结构

OSI（7层）：物理层，链路层，网络层，传输层，会话层，表示层，应用层（谐音：物联网谁会使用）

TCP/IP（4层）：链路层，网络层，传输层，应用层

链路层：设备之间的连接

网络层：主机之间

传输层：进程之间

# 2 UDP 编程

## 2.1 字节序

字节序是指**多字节数据**的存储顺序。

小端：高位字节数据存储在内存的高地址中，低位字节数据存储在内存的低地址中。

大端：高位字节数据存储在内存的低地址中，低位字节数据存储在内存的高地址中。**和我们的阅读习惯一致**

主机字节序和网络字节序：数据在网络中传输用的是大端（**网络字节序是大端**），而每个主机它的字节序因机器而异（主机字节序因机而异）。

```c
// 判断主机的字节序
bool isLittleEndian(){
    unsigned short a = 0x1218;//0x表示这个数按照16进制数识别，一个数字占4bit，所以1218占用2byte

    if( (*(char*)&a)  == 0x18){
        return true;
    }else{
        return false;
    }
}

#include<stdio.h>

typedef union Data {
	unsigned short a;
	char b[2];//b[0]和b[1]分别代表了a的两个字节
} data;
int main() {
	data tmp;
	tmp.a = 0x6141;// 16进制数的61等于十进制的97（a)，而16进制41是十进制的65（A）
	if (tmp.b[0] == 0x61) {
		printf("%c", tmp.b[0]);//打印出a，说明是0x61，就是大端，和我们的阅读习惯一致
	}
	else {
		printf("%c", tmp.b[0]);//打印出A，说明是0x41，就是小端，和我们的阅读习惯不一致
	}
	return 0;
}
```

### 2.1.1 字节序转换函数

头文件：`#include<arpa/inet.h>`

1. `uint32_t htonl(uint_t hostint32)`：host to net long（这个long，仅代表是4个byte（32位）的长度）
   - 功能：将32位主机字节序数据转换成网络字节序数据
   - 四个字节，一般用来转ip
2. `uint16_t htons(uint16_t hostint16)`：host to net short（这个short，仅代表是2个byte（16位）的长度）
   - 功能：将16位主机字节序数据转换成网络字节序数据
   - 一般用来转端口
3. `uint32_t ntohl(uint32_t netint32)`：net to host long
   - 功能：将32位网络字节序数据转换成主机字节序数据
4. `uint16_t ntohs(uint16_t netint16)`：net to host short





