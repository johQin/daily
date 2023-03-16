/* ************************************************************************
 *       Filename:  01tcp_client.c
 *    Description:  
 *        Version:  1.0
 *        Created:  2019年09月18日 14时12分31秒
 *       Revision:  none
 *       Compiler:  gcc
 *         Author:  YOUR NAME (), 
 *        Company:  
 * ************************************************************************/


#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <string.h>

int main(int argc, char *argv[])
{
	//创建流式套接字
	int s_fd = socket(AF_INET,SOCK_STREAM,0);
	if(s_fd < 0)
		perror("");
	//连接服务器
	struct sockaddr_in ser_addr;
	ser_addr.sin_family = AF_INET;
	ser_addr.sin_port = htons(8080);//服务器的端口
	ser_addr.sin_addr.s_addr = inet_addr("192.168.127.1");//服务器的ip
	int ret = connect(s_fd, (struct sockaddr*)&ser_addr,sizeof(ser_addr));
	if(ret<0)
		perror("");
	//收发数据
	while(1)
	{
		char buf[128]="";
		char r_buf[128]="";
		fgets(buf,sizeof(buf),stdin);
		buf[strlen(buf)-1]=0;
		write(s_fd,buf,strlen(buf));
		int cont = read(s_fd,r_buf,sizeof(r_buf));
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


