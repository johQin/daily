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



头文件：`#include<arpa/inet.h>`，windows则需要`#include <windows.h>`

在linux下，`/usr/include/arpa/inet.h`可以找到对应的文件。

1. `uint32_t htonl(uint_t hostint32)`：host to net long（这个long，仅代表是4个byte（32位）的长度）
   - 功能：将32位主机字节序数据转换成网络字节序数据
   - 四个字节，一般用来转ip
2. `uint16_t htons(uint16_t hostint16)`：host to net short（这个short，仅代表是2个byte（16位）的长度）
   - 功能：将16位主机字节序数据转换成网络字节序数据
   - 一般用来转端口
3. `uint32_t ntohl(uint32_t netint32)`：net to host long
   - 功能：将32位网络字节序数据转换成主机字节序数据
4. `uint16_t ntohs(uint16_t netint16)`：net to host short

```c
#include<stdio.h>
//linux 环境
//#include<arpa/inet.h>
//windows 环境
#include <windows.h>
#pragma comment(lib, "wsock32.lib")

int main() {
	int num4bytehost = 0x01020304;
	short num2bytehost = 0x0102;
	printf("%x\n", htonl(num4bytehost));
	printf("%x\n", htons(num2bytehost));
	return 0;
}
```



### 2.1.2 地址转换函数

点分十进制ip串转成整型的数据，便于将ip串放入分组包中。

1. `int inet_pton(int af,const char *src, void *dst)`

   - af：地址族address family（相当于协议族），AF_INET——ipv4
   - src：点分10进制串的首元素地址
   - dst：整型数据
   - 返回值：1——转换成功，其他值转换失败

2. `const char *inet_ntop(int af, const void *src, char *dst, socklen_t size)`

   - 功能：将网络大端32位数据转换为一个点分十进制

   - src：32网络大端的数据地址

   - dst：点分十进制

   - `#define INET_ADDRSTRLEN  16`  //for ipv4

     `#define INET6_ADDRSTRLEN 46` //for ipv6

3. 

```c
#include<stdio.h>
#include<arpa/inet.h>

int main() {
	char buf_ip[] = "192.168.1.2";
	int num = 0;
	inet_pton(AF_INET, buf_ip, &num);
	unsigned char* p = (unsigned char *)&num;
	printf("%d %d %d %d \n", *p, *(p + 1), *(p + 2), *(p + 3));

    #include<stdio.h>
#include<arpa/inet.h>

int main() {
	char buf_ip[] = "192.168.1.2";
	int num = 0;
	inet_pton(AF_INET, buf_ip, &num);
	unsigned char* p = (unsigned char *)&num;
	printf("%d %d %d %d \n", *p, *(p + 1), *(p + 2), *(p + 3));

	char ip[INET_ADDRSTRLEN] = "";
	printf("ip=%s\n", &num, ip, INET_ADDRSTRLEN);
    
	return 0;
}
```

## 2.2 UDP通信流程

**网络通信要解决的是不同主机进程间的通信**

socket提供不同主机上的进程之间的通信

- 每个套接字都有一个ip:port
- 对于UDP来讲，可以没有服务器，通常认定主动发送的一方为客户端，被动接收的一方为服务器。
- 如作为UDP服务器被动接收别人发送的数据，必须通过bind()指定ip:port，并且告知客户端，客户端的ip:port可以随机，
- 客户端发消息给服务器，会携带客户端ip和port，当UDP服务器给客户端，就可以从客户端发的数据当中获取客户端的ip:port
- 流程：创建套接字 -> 绑定（ip:port，非必须) -> 读写 -> 关闭

每一台主机必须要有和对方主机成对出现的一个socket，socket在不同的主机间必须成对存在（socket pair）

![](./legend/UDP_C_S架构图.png)

## 2.3 创建套接字

`#include <sys/socket.h>`

`int socket(int domain, int type, int protocol);`

- 功能：创建一个socket套接字
- domain：选择协议，一般AF_INET
- type：通信的语义
  - SOCKET_STREAM：流式套接字，用于TCP通信
  - SOCKET_DGRAM：报式套接字，用于UDP通信
- protocol：0，自动指定
- 返回值：返回一个套接字（文件描述符）



`#include <sys/socket.h>`

`int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen)`

- 功能：给套接字绑定一个固定的ip和端口

- 参数：

  - sockfd：套接字

  - addr：这里是一个通用套接字结构体，网络通信需要解决三大问题：协议，ip，port，可以把这三个数据封装在一个结构体中，

    - 为了使不同格式地址能被传入套接字函数,地址须要强制转换成通用套接字地址结构

      ```c
      #include <netinet/in.h>
      struct sockaddr
      {
      	sa_family_t sa_family;	// 2字节
      	char sa_data[14]	//14字节
      };
      ```

      

    - ipv4套接字结构体：\#include <netinet/in.h>

      ```c
      struct in_addr
      {
      	in_addr_t s_addr;//4字节
      };
      struct sockaddr_in
      {
      	sa_family_t sin_family;//2字节，AF_INET
      	in_port_t sin_port;//2字节，端口
      	struct in_addr sin_addr;//4字节，ip地址
      };
      
      ```

    - ipv6套接字结构体

    - 本地套接字结构体

- addrlen：INET_ADDRSTRLEN，值为16

- 返回值：1成功



`#include<sys/types.h>`

`#include <sys/socket.h>`

`ssize_t sendto(int socketfd, const void *buf, size_t len, int flags, const struct sockaddr *dest_addr, socklen_t addrlen)`

- buf：发送的内容地址
- len：发送的内容长度
- flags：0，
- dest_addr：ipv4套接字结构体（目的地址信息）
- 返回值：发送的字节数

`ssize_t recvfrom(int sockfd, void *buf, size_t len, int flags, struct sockaddr *src_addr, socklen_t *addrlen);`

- 返回值：将返回收到的字节数



### 2.3.1 客户端UDP编程

```c
#include<stdio.h>
#include<arpa/inet.h>
#include<sys/socket.h>
#include<string.h>
#include<unistd.h>
#include<sys/types.h>
int main(){
    //ipv4结构体
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_adder.sin_port = htons(9000);
    inet_pton(AF_INET,"192.168.127.1",&server_addr.sinaddr.s_addr);
    // 创建套接字
    int sock_fd = socket(AF_INET,SOCK_DGRAM,0);
    if(sock_fd<0){
        perror("");
    }
    while(1){
        // 获取发送的内容
        char buf[128]="";
        fgets(buf,sizeof(buf),stdin);
        buf[strlen[buf] - 1]=0;
        // 发送
        sendto(sock_fd,buf,strlen(buf),0,(struct sockaddr*) &server_addr,sizeof(server_addr));
        
        char read_buf[128]="";
        // 接收
        // 只要是设备文件，管道，网络，默认都会堵塞
        recfrom(sock_fd, read_buf,sizeof(read_buf),0,NULL,NULL);
        printf("%s\n", read_buf);
    }
}
```

### 2.3.2 服务器UDP编程

INADDR_ANY 通配地址，值为0，`#include<sys/socket.h>`

```c
#include<stdio.h>
#include<arpa/inet.h>
#include<sys/socket.h>
int main(){
    // 创建socket
    int sock_fd= socket(AF_INET,SOCK_DGRAM,0);
    if(sock_fd<0)perror("");
    
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(9090);
    //inet_pton(AF_INET,"192.168.127.129",&addr.sin_addr.s_addr);
    addr.sin_addr.s_addr = 0;// 通配地址，表示该主机的所有ip,#include<sys/socket.h>,INADDR_ANY 
    
    // 绑定
    int res = bind(sock_fd,(struct sockaddr*) &addr,sizeof(addr));
    if(res<0)perror("");
    
    struct sockaddr_in client_addr;
    socklen_t len = sizeof(client_addr);
    
    while(1){
        char buf[128]="";
        // 接收
        recvfrom(sock_fd, buf, sizeof(buf), 0, (struct sockaddr*) &client_addr, &len);
        printf("%s\n",buf);
        // 发送
        sendto(sock_fd, buf, n, 0, (struct sockaddr*)&client_addr, sizeof(client_addr));
    }
    close(sock_fd);
}
```

## 2.4 [TFTP](https://blog.csdn.net/qq_45740212/article/details/113034164)

TFTP（Trivial File Transfer Protocol）简单文件传输协议，一般用于局域网传输文件

提供不复杂、开销不大的文件传输服务，端口号为69。

特点：基于UDP，不会对用户进行验证（无需用户名和密码校验），而FTP需要校验

数据传输模式：

- octet：二进制模式
- netascii：文本模式
- mail：已经不再支持

### 2.4.1 TFTP通信过程

1. 服务器在69号端口等待客户端的请求
2. 服务器若批准此请求,则使用**临时端口**与客户端进行通信
3. 每个数据包的编号都有变化（从1开始）
4. 每个数据包都要得到ACK的确认如果出现超时,则需要重新发送最后的包（数据或ACK）
5. 数据的长度以512Byte传输
6. 小于512Byte的数据意味着传输结束

![](./legend/TFTP协议通信过程.png)

#### TFTP包格式

TFTP共定义了五种类型的包，包的类型由数据包前两个字节确定，我们称之为Opcode（操作码）字段。这五种类型的数据包分别是：

1. 读文件请求包：Read request，简写为RRQ，对应Opcode字段值为1，从服务器上下载
2. 写文件请求包：Write requst，简写为WRQ，对应Opcode字段值为2，上传文件到服务器
   - tsize选项
     - 当读操作时，tsize选项的参数必须为“0”，服务器会返回待读取的文件的大小
     - 当写操作时，tsize选项参数应为待写入文件的大小，服务器会回显该选项
   - blksize选项
     - 修改传输文件时使用的数据块的大小（范围：8～65464）
   - timeout选项
     - 修改默认的数据传输超时时间（单位：秒）
3. 文件数据包：Data，简写为DATA，对应Opcode字段值为3
4. 回应包：Acknowledgement，简写为ACK，对应Opcode字段值为4，如果读写请求包带有选项，将会返回OACK包
5. 错误信息包：Error，简写为ERROR，对应Opcode字段值为5
   - 未定义,参见错误信息
   - File not found.
   - Access violation.
   -  Disk full or allocation exceeded.
   - illegal TFTP operation.
   - Unknown transfer ID
   -  File already exists.
   -  No such user.
   - Unsupported option(s) requested.

![](./legend/TFTP包示意图.png)

TFTP下载请求流程：

1. 客户端创建一个套接字
2. 客户端打开或创建一个文件
3. 客户端发送一个下载请求：sprintf()
4. 循环接收数据，存在文件中

```c
//下载
void tftp_down(char *argv)
{
	int fd;
	unsigned short p_num = 0;
	unsigned char cmd = 0;
	char cmd_buf[512] = "";
	char recv_buf[516] = "";
	struct sockaddr_in client_addr;
	socklen_t cliaddr_len = sizeof(client_addr);

	if(dest_addr.sin_port == 0){
		dest_addr.sin_family = AF_INET;
		dest_addr.sin_port = htons(69); 
		puts("send to IP:");
		fgets(recv_buf,sizeof(recv_buf),stdin);
		*(strchr(recv_buf,'\n')) = '\0';
		inet_pton(AF_INET, recv_buf, &dest_addr.sin_addr);
	}
	
	//构造下载请求,argv为文件名
	int len = sprintf(cmd_buf, "%c%c%s%c%s%c", 0, 1, argv, 0, "octet", 0);	//发送读数据包请求
	sendto(sockfd, cmd_buf, len, 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
	
	fd = open(argv, O_WRONLY|O_CREAT, 0666);
	if(fd < 0 ){
		perror("open error");
		close(sockfd);
		exit(-1);
	}
	
	do{
		//接收服务器发送的内容
		len = recvfrom(sockfd, recv_buf, sizeof(recv_buf), 0, (struct sockaddr*)&client_addr, &cliaddr_len);
		
		cmd = recv_buf[1];
		if( cmd == 3 )	//是否为数据包
		{
			//接收的包编号是否为上次包编号+1
			if((unsigned short)(p_num+1) == ntohs(*(unsigned short*)(recv_buf+2) ))
			{
				write(fd, recv_buf+4, len-4);
				p_num = ntohs(*(unsigned short*)(recv_buf+2));
				
				printf("recv:%d\n", p_num);//十进制方式打印包编号
			}			
			recv_buf[1] = 4;
			sendto(sockfd, recv_buf, 4, 0, (struct sockaddr*)&client_addr, sizeof(client_addr));
		}
		else if( cmd == 5 ) //是否为错误应答
		{
			close(sockfd);
			close(fd);
			unlink(argv);//删除文件
			printf("error:%s\n", recv_buf+4);
			exit(-1);
		}		
	}while((len == 516)||(cmd == 6)); //如果收到的数据小于516则认为出错
	close(fd);
	PRINT("Download File is Successful\n", RED);
	return;
}
```



## 2.5 UDP广播

广播：由一台主机向该主机所在子网内的所有主机发送数据的方式，不能发送到另一个子网。

广播只能用UDP或原始套接字实现，不能用TCP。

基于UDP的广播协议：

1. 地址解析协议（ARP）：是根据IP地址获取物理地址的一个TCP/IP协议。RARP由物理地址获取ip
2. 动态主机配置协议（DHCP）：向路由器申请一个ip。需要IP地址的主机在启动时就向DHCP服务器广播发现报文。
3. 网络时间协议（NTP）

广播的特点：

1. 处于同一子网的所有主机都必须处理数据
2. UDP数据包会沿着协议栈一直向上至传输（UDP)层
   - 链路层，广播mac地址全f，地址：ff:ff:ff:ff:ff:ff
   - 在网络层，ip地址 = < 网络号，主机号>，路由器隔离广播域，不转发广播数据包
     - **定向广播地址**，主机号ip全1，地址：192.168.252.255/24，
     - **受限广播地址**，网络号和主机号全1，地址：255.255.255.255，
   - 所以这个数据包，网络层和链路层都会被处理
3. 运行音视频等较高速率工作的应用，会带来大的负荷
4. 局限于同一子网

![](./legend/UDP广播.png)

### UDP广播编程

`int setsockopt(int sockfd, int level, int optname, const void *optval, socklen_t optlen);`

`int getsockopt(int sockfd, int level, int optname, void *optval, socklen_t *optlen);`

1. int sockfd: 很简单，socket句柄
2. int level: 选项定义的层次；目前仅支持SOL_SOCKET和IPPROTO_TCP层次
3. int optname: 需设置的选项
   - level=SOL_SOCKET时，可设置的optname有
     - SO_BROADCAST
     - SO_RCVBUF
     - SO_SNDBUF 
     - 等等
4. const void *optval: 指针，指向存放选项值的缓冲区
5. socklen_t optlen: optval缓冲区的长度



设置套接字有广播功能：

- sock：套接字
- level：SOL_SOCKET
- optname：SO_BROADCAST，默认6
- optval：int类型变量的地址，这个值设置为1，
- optlen：optval类型的大小

```c
int main(int argc,char *argv[]){
    int sock_fd=0;
    char buff[1024]="";
    unsigned short port = 8000;
    struct sockaddr_in send_addr;
    
    bzero (&send_addr, sizeof(send_addr));
    send_addr.sin_family = AF_INET;
    send_addr.sin_port = htons(port);
    sock_fd = socket(AF_INET, SOCK_DGRAM,0);
    
    if(sock_fd<0){
		perror("socket failed");
        close(sock_fd);
        exit(1);
    }
    
    if(argc >1){
        send_addr.sin_addr.s_addr = inet_addr(argv[1]);
    }else{
        printf("not have a server ip");
        exit(1);
    }
    
    int yes = 1;
    setsockopt(sock_fd,SOL_SOCKET,SO_BROADCAST, &yes, sizeof(yes));
    strcpy(buff,"broadcast success");
    int len = sendto(sock_fd,buff,strlen(buff),0,(struct sockaddr *) &send_addr, sizeof(send_addr));
    if(len<0){
        printf("send error\n");
        close(sock_fd);
        exit(1)
    }
    return 0;
}
```

## 2.6 UDP多播

数据的收发仅仅在同一分组中进行。

多播的特点：

- 一个多播地址（D类地址）标识一个多播分组，一个分组内的所有主机，都有一个相同的多播ip地址

- 多播可以用于广域网使用

  | 类型 | ipv4 | ipv6   |
  | ---- | ---- | ------ |
  | 单播 | 支持 | 支持   |
  | 多播 | 可选 | 支持   |
  | 广播 | 支持 | 不支持 |
  | 任播 | 支持 | 支持   |

多播ip地址：D类IP，规定是224.0.0.0-239.255.255.255。

- 224.0.0.0：基准地址，保留
- 224.0.0.1：表示组内所有主机
- 其他的多播地址有的有特殊的含义。

多播mac地址：

- 多播mac地址是第一个字节的最低位为1的所有地址，例如01-12-0f-00-00-02。
- 后三个字节一般是多播ip地址的映射，01-12-0f-**00-00-01**，加粗部分是ip后三个点分十进制的映射。

所有加入多播组的主机都会得到一个同样的多播ip地址，同时根据多播ip地址生成一个临时的多播mac地址。

### 多播编程

在IPv4因特网域(AF_INET)中，多播地址结构体用如下结构体ip_mreq表示

```c
struct in_addr{
    in_addr_t s_addr;
}
struct ip_mreq{
    struct in_addr imr_multiaddr;//多播组ip地址
    struct in_addr imr_interface;//将要添加到多播组的主机ip
}
```

多播套接字选项

`int setsockopt(int sockfd, int level, int optname, const void *optval, socklen_t optlen);`

- | level      | optname            | 说明       | optval类型 |
  | ---------- | ------------------ | ---------- | ---------- |
  | IPPROTO_IP | IP_ADD_MEMBERSHIP  | 加入多播组 | ip_mreq{}  |
  | IPPROTO_IP | IP_DROP_MEMBERSHIP | 离开多播组 | ip_mreq{}  |

```c
char group[INET_ADDRSTRLEN]= "224.0.0.1";

struct ip_mreq mreq;
mreq.imr_multiaddr.s_addr = inet_addr(group);
mreq.imr_interface.s_addr = honl(INADDR_ANY);//INADDR_ANY,全0，表示网络本身，

setsockopt(sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq));
```

# 3 TCP 编程



![](./legend/TCP_C_S架构.png)

## 3.1 创建套接字

### 3.1.1 客户端

`int socket(int domain, int type, int protocol);`

- 功能：创建一个socket套接字
- domain：选择协议，一般AF_INET
- type：通信的语义
  - SOCKET_STREAM：流式套接字，用于TCP通信
  - SOCKET_DGRAM：报式套接字，用于UDP通信
- protocol：0，自动指定
- 返回值：返回一个套接字（文件描述符）



`int connect(int sockfd, const struct sockaddr *addr,socklen_t addrlen);`

- 功能：连接服务器
- 参数：
  - sockfd：套接字
  - addr：服务器地址信息
  - addrlen：addr结构体大小
- 返回值：0——成功，-1——失败，

```c
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <string.h>

int main(int argc, char *argv[])
{
	//创建流式套接字
	int s_fd = socket(AF_INET,SOCK_STREAM,0);
	if(s_fd < 0) perror("");
    
	//连接服务器
	struct sockaddr_in ser_addr;
	ser_addr.sin_family = AF_INET;
	ser_addr.sin_port = htons(8080);//服务器的端口
	ser_addr.sin_addr.s_addr = inet_addr("192.168.127.1");//服务器的ip
	int ret = connect(s_fd, (struct sockaddr*)&ser_addr,sizeof(ser_addr));
	if(ret<0) perror("");
    
	//收发数据
	while(1)
	{
		char buf[128]="";
		char r_buf[128]="";
		fgets(buf,sizeof(buf),stdin);
		buf[strlen(buf)-1]=0;
        //发
		write(s_fd,buf,strlen(buf));
		//收
        int cont = read(s_fd,r_buf,sizeof(r_buf));
        
        ////如果对方挂断，会发送一个0长度的数据报，此时cont为0
		if(cont == 0)
		{
			printf("server close\n");
			break;//对方关闭
		}
		printf("recv server = %s\n",r_buf);
	
	}
	//关闭套接字'
	close(s_fd);
	return 0;
}
```

### 3.1.2 服务器

1. socket()：创建套接字
2. bind()：绑定ip和端口
3. listen()：监听
   - `int listen(int sockfd, int backlog);`
   - 功能：
     - 套接字由主动变为被动（**监听套接字，lfd**）
     - 创建两个连接队列，一个已完成连接队列（完成三次握手），一个未完成连接队列
   - 参数：
     - backlog：两个连接对列的连接数目之和最大值
4. accept()：提取
   - `int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);`
   - 功能：
     - 从已完成连接队列里提取一个新的连接，
     - 然后**创建一个新的已连接套接字（通信套接字，cfd）**，
     - 使用这个已连接套接字和当前连接的客户端通信
   - 参数：
     - addr保存客户端地址的信息结构体
   - 返回值：成功返回已连接套接字
5. read()/write()
6. close()

**accept和read是带阻塞的。无法在一个服务器上，同时和多个客户端通信**

**当accept和read被信号（CTRL + C）中断后，他们将不会再阻塞**

![](./legend/TCP服务器工作示意图.png)

```c
// 处理单个客户端
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>
int main(int argc, char *argv[])
{
	//创建套接字，监听套接字
	int lfd = socket(AF_INET,SOCK_STREAM,0);
	if(lfd < 0)
		perror("");
	//绑定
	struct sockaddr_in myaddr;
	myaddr.sin_family = AF_INET;
	myaddr.sin_port = htons(9999);
	myaddr.sin_addr.s_addr = 0;//本机的ip都绑定
	bind(lfd,(struct sockaddr*)&myaddr,sizeof(myaddr));//绑定
	//监听
	listen(lfd,128);
	//提取
	struct sockaddr_in cliaddr;
	socklen_t len = sizeof(cliaddr);
	
	// 通信套接字
	int cfd = accept(lfd,(struct sockaddr *)&cliaddr,&len);//提取
	if(cfd < 0 )
		perror("");
	char ip[16]="";
	printf("client ip=%s port=%d\n",
			inet_ntop(AF_INET,&cliaddr.sin_addr.s_addr,ip,16),ntohs(cliaddr.sin_port));
	//读写
	while(1)
	{
		char buf[256]="";
		//回射服务器
		int ret = read(cfd,buf,sizeof(buf));
		if(ret == 0)
		{
			printf("client close\n");
			break;
		}
		printf("recv = [%s]\n",buf);
		write(cfd,buf,ret);
	
	}
	//关闭
	close(lfd);
	close(cfd);
	return 0;
}
```

## 3.2 并发服务器

### 3.2.1 多进程实现

![](./legend/多进程tcp并发通信.png)

```c
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>

//回收子进程资源
void cath_child(int num)
{
	pid_t pid;
	while(1)
	{
		pid = waitpid(-1,NULL,WNOHANG);
		if(pid <= 0)
		{
            //pid = -1，所有子进程都回收了
            //pid = 0，还有子进程活着
			break;
		}
		else if(pid > 0)
		{
            //继续回收，某个子进程被回收了
			printf("child process %d\n",pid);
			continue;
		}
	}

}
int main(int argc, char *argv[])
{
	//将SIGCHLD 加入到阻塞集中
	sigset_t set;
	sigemptyset(&set);
	sigaddset(&set,SIGCHLD);
	sigprocmask(SIG_BLOCK,&set,NULL);
    
    
	// 创建套接字
	int lfd = socket(AF_INET,SOCK_STREAM,0);
	if(lfd <0) perror("");
	// 绑定
	struct sockaddr_in myaddr;
	myaddr.sin_family = AF_INET;
	myaddr.sin_port = htons(6666);
	myaddr.sin_addr.s_addr = 0;
	bind(lfd,(struct sockaddr *)&myaddr,sizeof(myaddr));
	// 监听
	listen(lfd,128);
	// 提取
	struct sockaddr_in cliaddr;
	socklen_t len = sizeof(cliaddr);
	while(1)
	{
        //当accept和read被信号（CTRL + C）中断后，他们将不会再阻塞,所以这里有点问题
        // 当父进程被Ctrl +c后，accept不会阻塞，程序向下进行，将会出问题。
		int cfd = accept(lfd,(struct sockaddr*)&cliaddr,&len);
		if(cfd < 0)perror("");
        
		char ip[16]="";
        
        //打印客户端相关信息
		printf("client ip=%s port=%d\n",
				inet_ntop(AF_INET,&cliaddr.sin_addr.s_addr,ip,16),
				ntohs(cliaddr.sin_port));
        
     	// 创建子进程
		pid_t pid;
		pid = fork();
		if(pid ==0 )//子进程
		{
			close(lfd);
			while(1){
				sleep(1);
				char buf[256]="";
				int n = read(cfd,buf,sizeof(buf));
				// 通信结束子进程退出
                if(0==n){
					printf("client close\n");
					break;
				}
				printf("[***%s***]\n",buf);
				write(cfd,buf,n);
			}
			close(cfd);
			exit(0);
		}
        
        // 父进程
		else if(pid > 0)
		{
			close(cfd);
			struct sigaction act;
			act.sa_handler = cath_child;
			act.sa_flags = 0;
			sigemptyset(&act.sa_mask);
			sigaction(SIGCHLD,&act,NULL);

			sigprocmask(SIG_UNBLOCK,&set,NULL);
		}
	 }
	

	return 0;
}
```



### 3.2.2 多线程实现

```c
#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>

#include <stdlib.h>
typedef struct _info
{
	int cfd;
	struct sockaddr_in client;

}INFO;

// 与客户端通信
void* resq_client(void *arg)
{
	INFO *info = (INFO*)arg;
	char ip[16]="";
	printf("client ip=%s port=%d\n",
			inet_ntop(AF_INET,&(info->client.sin_addr.s_addr),ip,16),
			ntohs(info->client.sin_port));
	while(1)
	{
		char buf[1024]="";
		int n = read(info->cfd,buf,sizeof(buf));
		if(n == 0)
		{
			printf("cleint close\n");
			break;
		}
		printf("%s\n",buf);
		write(info->cfd,buf,n);
	}
	close(info->cfd);
	free(info);
}
int main(int argc, char *argv[])
{
	int lfd = socket(AF_INET,SOCK_STREAM,0);
	if(lfd <0)
		perror("");

	int opt=1;
	setsockopt(lfd,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));

	struct sockaddr_in myaddr;
	myaddr.sin_family = AF_INET	;
	myaddr.sin_port= htons(8888);
	myaddr.sin_addr.s_addr = 0;
	if( bind(lfd,(struct sockaddr*)&myaddr,sizeof(myaddr))< 0)
		perror("bind:");


	listen(lfd,128);
	while(1)
	{
		struct sockaddr_in client;
		socklen_t len = sizeof(client);
		int cfd = accept(lfd,(struct sockaddr*)&client,&len);
		INFO *info = malloc(sizeof(INFO));
		info->cfd =cfd;
		info->client = client;
		pthread_t pthid;
		pthread_create(&pthid,NULL,resq_client,info);
		pthread_detach(pthid);
	}

	close(lfd);
	return 0;
}
```



### 3.2.3 非阻塞忙轮询

多线程和多进程的方式，对资源的消耗是巨大的。来一个客户端请求，服务器就创建一个进程或线程处理，这种方式实现并发，大多数时间，我们的进程和线程都处于休眠状态（阻塞，阻塞等待（accept和read默认都是带阻塞的）），这样非常浪费资源。

由服务器进程不停的访问每个套接字，查看是否有连接或数据到达。

这样的方式非常浪费cpu。

### 3.2.4 IO多路复用

服务器单进程 + 内核监听请求（多个fd）的方式

IO多路复用：由内核监听多个文件描述符的读写缓冲区的变化（不仅仅能监听网络的fd，管道的，stdin/stdout/stderr，任何设备的都可以进行监听），一旦有变化，就会让服务器进程来处理。

在这里，监听的文件描述符分两类，一个是lfd，一个是cfd

进程A执行到创建socket语句的时候，创建socket对象，进程A进入阻塞态，等待接收socket传来的网络数据，当socket已接收到数据之后，进程A进入执行态

![](./legend/进程与单个socket.png)

IO多路复用有两种方式：[select和epoll](https://www.jianshu.com/p/c9190109c7d8)



### 3.2.5 select

select准备一个数组fds（文件描述符），存放需要监视的所有socket，然后调用select，如果fds中所有的socket都没有数据，select会阻塞，直到有一个socket收到数据，select返回，唤醒进程，用户可以遍历fds，通过FD_ISSET判断哪个socket收到了数据，然后做出处理。windows下面用的最多的就是select

![](./legend/select.png)

 

       #include <sys/select.h>
       #include <sys/time.h>
       #include <sys/types.h>
       #include <unistd.h>

`int select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);`

- 函数功能：监听多个文件描述符的属性变化（读写缓冲区的变化）

- 参数：

  - nfds：需要监听的文件描述符的个数 +1（n个cfd + lfd），最大支持FD_SETSIZE=1024

  - readfds：需要监听读属性变化的文件描述符集合

  - writefds：需要监听写属性变化的文件描述符集合

  - exceptfds：需要监听异常属性变化的文件描述符集合

  - timeout：超时时间，表示等待多长时间之后就放弃等待，

    - 传 NULL 表示等待无限长的时间，持续阻塞直到有事件就绪才返回。
    - 大于0，超时时间
    - =0，不等待立即返回

    ```c
    struct timeval{
        long tv_sec;//秒
        long tv_usec;//微秒
    }
    ```

- 返回值：变化的文件描述符个数

内核在监听到readfds有fd读属性变化的时候，就会将readfds集合重置为变化的fd的集合（而不是一开始的需要监听的fd集合），所以我们需要对初始需要监听的fd集合做一个备份，以便下一次继续监听。

文件描述符集合操作函数：

- `void FD_CLR(int fd, fd_set *set);`：将fd从集合set中剔除
- ``int  FD_ISSET(int fd, fd_set *set);`：判断fd是否在set里面
- `void FD_SET(int fd, fd_set *set);`：添加fd到set里面
- `void FD_ZERO(fd_set *set);`：清空set集合

优点：

- 可跨平台，windows，linux都可以
- 并发量高，效率高，消耗资源少

缺点：

- 最大监听fd数目为1024
- 每一次监听都需要再次设置需要监听的集合，集合从用户态拷贝至内核态消耗资源
- 监听到fd变化后，需要用户自己遍历集合，才知道具体哪个fd发生了变化
- 大量并发，少数活跃，select效率低

#### 代码实现

```c
#include <stdio.h>
#include "wrap.h"
#include <sys/time.h>
#include <sys/select.h>
#include <sys/types.h>
int main(int argc, char *argv[])
{
	//创建套接字
	//绑定
	int lfd = tcp4bind(9999,NULL);//在wrap.h中
	//监听
	listen(lfd,128);
	int max_fd = lfd;
	fd_set r_set;
	fd_set old_set;
	FD_ZERO(&old_set);
	FD_ZERO(&r_set);

	FD_SET(lfd,&old_set);
	int nready=0;
	while(1)
	{
		r_set = old_set;
		nready = select(max_fd+1,&r_set,NULL,NULL,NULL);
		if(nready < 0)
		{		
			perror("");
			break;
		}
		else if(nready >= 0)
		{
			//判断lfd是否变化了，如果变化生成一个新的cfd
			if(FD_ISSET(lfd,&r_set))
			{
				// 提取新的连接
				struct sockaddr_in cliaddr;
				socklen_t len=sizeof(cliaddr);
				char ip[16]="";
				int cfd = Accept(lfd,(struct sockaddr*)&cliaddr,&len);
				printf("client ip=%s port=%d\n",
						inet_ntop(AF_INET,&cliaddr.sin_addr.s_addr,ip,16),
						ntohs(cliaddr.sin_port));
				//将cfd加入到old_set
				FD_SET(cfd,&old_set);
				//更新最大值
				if(max_fd < cfd)
				max_fd = cfd;
				//如果只有lfd变化,执行下一次监听
				if( --nready == 0)
					continue;
			
			}
			for(int i=lfd+1;i<=max_fd;i++)
			{
				//cfd变化
				if(FD_ISSET(i,&r_set))
				{
					char buf[1024]="";
					int n = Read(i,buf,sizeof(buf));
					if(n == 0)
					{
						printf("client close\n");
						close(i);
						//将i从old_set删除
						FD_CLR(i,&old_set);
					}
					else if(n < 0)
					{
						perror("");
					
					}
					else
					{
						printf("%s\n",buf);
						Write(i,buf,n);
					}
				}
				
			
			
			}
		}
	
	}
	return 0;
}

```

#### select优化

主要是在查找哪个fd变化上（遍历fds）上做优化

### 3.2.6 epoll

epoll创建epoll对象，维护一个就绪列表和等待队列。当socket接收到数据，中断程序一方面修改rdlist，另一方面唤醒eventpoll等待队列中的进程，由于rdlist的存在，进程A可以知道哪些socket发生了变化。

![](./legend/epoll.png)



`#include <sys/epoll.h>`

`int epoll_create(int size);`

- 功能：创建一个红黑树，用于管理需要监听的fds
- 参数：size大于0即可，容量不够时，函数会自动扩容
- 返回值：树的句柄

`int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);`

- 功能：对树节点做操作

- 参数：

  - epfd：树的根节点（句柄）

  - op：

    - EPOLL_CTL_ADD：添加fd
    - EPOLL_CTL_MOD：修改fd
    - EPOLL_CTL_DEL：删除fd

  - fd：需要操作的fd

  - event：树上的节点

    ```c
    typedef union epoll_data{
        void *ptr;
        int fd;//需要监听的fd
        unint32_t u32;
        unint64_t u64;
    } epoll_data_t;
    struct epoll_event{
        unint32_t events;//监听fd的什么事件，EPOLLIN读事件，EPOLLOUT写事件，EPOLLET,EPOLLLT,EPOLLRDHUP，EPOLLPRI，等等
        epoll_data_t data;
    }
    ```

- 返回值：操作成功0，失败-1

`int epoll_wait(int epfd, struct epoll_event *events,int maxevents, int timeout);`

- 功能：开启监听树上所有节点的属性变化
- 参数：
  - epfd：树的句柄
  - epoll_events：存放变化的fd的数组
  - maxevents：epoll_events的容量
  - timeout：永久监听（-1），限时等待（>0）
- 返回值：变化的fd个数

select与epoll的区别

- 每次调用select，都需要把fd集合从用户态拷贝到内核态，这个开销在fd很多时会很大；而epoll保证了每个fd在整个过程中只会拷贝一次。
- 每调用一次select都会遍历一次fd集合，知晓哪个有数据，而epoll只会轮询一次fd集合，查看rdlist
- select最大支持1024个fd，而epoll没有这个限制

## 3.3 其他概念

### 3.3.1 netstat

- **跟踪网络，查看进程对外和对内通信情况。**输出分两部分，分别是网络与系统进程相关性部分

- **netstat -[ atunlp ]**

- a—列出系统当前所有的连接、监听、sockete数据。（包括监听与未监听）

- t—列出tcp连接，u—列出udp连接

- n—不列进程的服务名称，以端口号显示，会影响Local Adress分号后面是端口号还是服务名

- l—列出正在网络监听的服务

```bash
# 列出系统当前所有的连接、监听、sockete数据。（包括监听与未监听）
netstat -a # a——all
# 列出所有TCP端口
netstat -at
# 列出所有UDP端口
netstat -au

# 列出所有处于监听状态的Sockets
netstat -l
netstat -lt
netstat -lu
netstat -lx

# 列出每个协议的统计信息
netstat -s
netstat -st
netstat -su

# 显示pid和进程名称
netstat -p 

```

### 3.3.2 地址复用

服务器启动后，谁主动关闭，谁就需要等待2MSL，服务占据的端口在2MSL之后，端口才会被释放。

如果在2MSL时间内，再次启动服务器，将会绑定失败，因为端口还未被释放。

`bind: address  already in use`

地址重用：SO_REUSEADDR

```c
int opt = 1;
setsocket(listenfd, SOL_SOCKET, SO_REUSEADDR, &opt,sizeof(opt));
//要在bind之前
```

注意：重新使用这个端口之后，原来的那个程序就不能用了

### 3.3.3 半关闭

半关闭：一边关闭（第一次挥手客户端发FIN，第二次挥手服务器发ACK，两次挥手之间），这时候处于FIN_WAIT_2状态，这个时候客户端只能收不能发。

如果客户端调用close只会关闭写（发），这时候读端还可以读数据，如果在收到服务器的FIN和ACK（第三次挥手服务器向客户端发送FIN，第四次挥手客户端回ACK时），这个时候才全部关闭。这里的半关闭状态是底层帮我们实现的，我们能不能在应用层写一个只能收或只能发的通信？

应用层实现半关闭状态：
`int shutdown(int sockfd,int how);`

- how:
  - SHUT_RD，关闭读操作，不能接收数据
  - SHUT_WR，关闭写操作，不能发数据
  - SHUT_RDWR，关闭读写操作，相当于调用了两次shutdown，首先是以SHUT_RD，然后SHUT_WR

### 3.3.4 心跳包

```c
int keepalive = 1;
setsocketopt(listenfd, SOL_SOCKET, SO_KEEPALIVE, (void *) &keepAlive,sizeof(keepAlive));
```

SO_KEEPALIVE保持连接，检测对方主机是否崩溃，避免服务器永远阻塞于TCP连接的输入，设置此选项后，如果两小时内此套接口的任一一方都没有数据交换，就会发出探测包，检测对方是否还正常。

**由于两小时时间太长，基本没什么应用场景。**

一般心跳包，由程序自行实现。

乒乓包：也是心跳包，携带的信息比较多



