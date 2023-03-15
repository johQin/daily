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

