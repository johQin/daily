/* ************************************************************************
 *       Filename:  02tcp_server.c
 *    Description:  
 *        Version:  1.0
 *        Created:  2019年09月18日 15时27分07秒
 *       Revision:  none
 *       Compiler:  gcc
 *         Author:  YOUR NAME (), 
 *        Company:  
 * ************************************************************************/


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


