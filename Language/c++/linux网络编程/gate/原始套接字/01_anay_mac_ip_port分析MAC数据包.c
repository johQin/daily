/* ************************************************************************
 *       Filename:  01_anay_mac_ip_port.c
 *    Description:  
 *        Version:  1.0
 *        Created:  2019年09月23日 14时02分18秒
 *       Revision:  none
 *       Compiler:  gcc
 *         Author:  贺溯  
 *        Company:  
 * ************************************************************************/


#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <string.h>
#include <netinet/in.h>
#include <netinet/ether.h>
int main(int argc, char *argv[])
{
	int fd = socket(PF_PACKET,SOCK_RAW,htons(ETH_P_ALL));
	if(fd <0)
		perror("");
	unsigned char buf[1500]="";
	unsigned char src_mac[18]="";
	unsigned char dst_mac[18]="";
	unsigned char dst_ip[16]="";
	unsigned char src_ip[16]="";
	

	while(1)
	{
		bzero(buf,sizeof(buf));
		bzero(src_mac,sizeof(src_mac));
		bzero(dst_mac,sizeof(dst_mac));
		recvfrom(fd,buf,sizeof(buf),0,NULL,NULL);
		//buf
		sprintf(dst_mac,"%x:%x:%x:%x:%x:%x",buf[0],buf[1],buf[2],buf[3],buf[4],buf[5]);
		sprintf(src_mac,"%x:%x:%x:%x:%x:%x",buf[6],buf[7],buf[8],buf[9],buf[10],buf[11]);
		printf("src_mac=%s --> dst_mac=%s\n",src_mac,dst_mac);
		if(buf[12]==0x08 && buf[13]==0x00)
		{
			printf("IP\n");
			sprintf(src_ip,"%d.%d.%d.%d",buf[26],buf[27],buf[28],buf[29]);
			sprintf(dst_ip,"%d.%d.%d.%d",buf[30],buf[31],buf[32],buf[33]);
			printf("src_ip=%s --> dst_ip=%s\n",src_ip,dst_ip);
			if( buf[23] == 6)
			{
				printf("TCP\n");
				printf("src_port=%d\n", ntohs(*(unsigned short*)(buf+34)));
				printf("dst_port=%d\n", ntohs(*(unsigned short*)(buf+36)));
			
			}
			else if( buf[23] == 17)
			{
				printf("UDP\n");
			
			}
		
		}
		else if(buf[12]==0x08 && buf[13]==0x06)
		{
			printf("ARP\n");
		
		}
		else if(buf[12]==0x80 && buf[13]==0x35)
		{
			printf("RARP\n");
			
		
		}
		
	
	
	}


	return 0;
}



