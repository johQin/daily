#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <net/if.h>				//struct ifreq
#include <sys/ioctl.h>			//ioctl、SIOCGIFADDR
#include <sys/socket.h>
#include <netinet/ether.h>		//ETH_P_ALL
#include <netpacket/packet.h>	//struct sockaddr_ll

#define IPMSG_BR_ENTRY 	1
#define IPMSG_BR_EXIT 	2
#define IPMSG_SENDMSG 	32
#define IPMSG_RECVMSG 	33


unsigned short checksum(unsigned short *buf, int len)
{
	int nword = len/2;
	unsigned long sum;

	if(len%2 == 1)
		nword++;
	for(sum = 0; nword > 0; nword--)
	{
		sum += *buf;
		buf++;
	}
	sum = (sum>>16) + (sum&0xffff);
	sum += (sum>>16);
	return ~sum;
}
int main(int argc, char *argv[])
{
	//1.创建通信用的原始套接字
	int sock_raw_fd = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
	
	//2.根据各种协议首部格式构建发送数据报
	unsigned char send_msg[1024] = {
		//--------------组MAC--------14------
		0x00, 0x21, 0xcc, 0x5f, 0x7e, 0xf6, //dst_mac
		0x00, 0x53, 0x50, 0x00, 0x4a, 0x44, //src_mac
		0x08, 0x00,                         //类型：0x0800 IP协议
		//--------------组IP---------20------
		0x45, 0x00, 0x00, 0x00,             	//版本号：4, 首部长度：20字节, TOS:0, --总长度--：
		0x00, 0x00, 0x00, 0x00,				//16位标识、3位标志、13位片偏移都设置0
		0x80, 17,   0x00, 0x00,				//TTL：128、协议：UDP（17）、16位首部校验和
		10,  0,   31,  17,				//src_ip
		10,  0,   31,  39,				//dst_ip
		//--------------组UDP--------8+len------
		0x09, 0x79, 0x09, 0x79,             //src_port:0x0979(2425), dst_port:0x0979(2425)
		0x00, 0x00, 0x00, 0x00,               //#--16位UDP长度--30个字节、#16位校验和
	};
	
	int len = sprintf(send_msg+42, 
		"1:%d:%s:%s:%d:%s", 	/* 版本头像 */
		123,	/* 飞秋数据包编号 */
		"usernam", 	/* 发送者姓名 */
		"hostname",	/* 发送者主机名 */
		IPMSG_SENDMSG,	/* 表示发送消息的命令 */
		"hello");	/* 发送内容 */
	if(len % 2 == 1)//判断len是否为奇数
		len++;//如果是奇数，len就应该加1(因为UDP的数据部分如果不为偶数需要用0填补)
	
	*((unsigned short *)&send_msg[16]) = htons(20+8+len);//IP总长度 = 20 + 8 + len
	*((unsigned short *)&send_msg[14+20+4]) = htons(8+len);//udp总长度 = 8 + len
	//3.UDP伪头部
	unsigned char pseudo_head[1024] = {
		//------------UDP伪头部--------12--
		10,  0,   31,  17,				//src_ip
		10,  0,   31,  39,				//dst_ip
		0x00, 17,   0x00, 0x08,             	//0,17,#--16位UDP长度--8个字节
	};
	
	*((unsigned short *)&pseudo_head[10]) = htons(8 + len);//伪头部中的udp长度（和真实udp长度是同一个值）
	//4.构建udp校验和需要的数据报 = udp伪头部 + udp数据报
	memcpy(pseudo_head+12, send_msg+34, 8+len);//--计算udp校验和时需要加上伪头部--
	//5.对IP首部进行校验
	*((unsigned short *)&send_msg[24]) = checksum((unsigned short *)(send_msg+14),20);
	//6.--对UDP数据进行校验--
	*((unsigned short *)&send_msg[40]) = checksum((unsigned short *)pseudo_head,12+8+len);
	
	
	//6.发送数据
	struct sockaddr_ll sll;					//原始套接字地址结构
	struct ifreq ethreq;					//网络接口地址
	
	strncpy(ethreq.ifr_name, "eth0", IFNAMSIZ);			//指定网卡名称
	if(-1 == ioctl(sock_raw_fd, SIOCGIFINDEX, &ethreq))	//获取网络接口
	{
		perror("ioctl");
		close(sock_raw_fd);
		exit(-1);
	}
	
	/*将网络接口赋值给原始套接字地址结构*/
	bzero(&sll, sizeof(sll));
	sll.sll_ifindex = ethreq.ifr_ifindex;
	len = sendto(sock_raw_fd, send_msg, 14+20+8+len, 0 , (struct sockaddr *)&sll, sizeof(sll));
	if(len == -1)
	{
		perror("sendto");
	}
	close(sock_raw_fd);
	return 0;
}



