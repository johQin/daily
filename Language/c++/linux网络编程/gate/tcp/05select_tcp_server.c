/* ************************************************************************
 *       Filename:  05select_tcp_server.c
 *    Description:  
 *        Version:  1.0
 *        Created:  2019年09月19日 17时19分56秒
 *       Revision:  none
 *       Compiler:  gcc
 *         Author:  YOUR NAME (), 
 *        Company:  
 * ************************************************************************/


#include <stdio.h>
#include "wrap.h"
#include <sys/time.h>
#include <sys/select.h>
#include <sys/types.h>
int main(int argc, char *argv[])
{
	//创建套接字
	//绑定
	int lfd = tcp4bind(9999,NULL);
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
			//判断lfd是否变化了
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


