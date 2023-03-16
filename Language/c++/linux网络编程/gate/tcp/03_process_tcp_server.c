/* ************************************************************************
 *       Filename:  03_process_tcp_server.c
 *    Description:  
 *        Version:  1.0
 *        Created:  2019年09月18日 16时36分29秒
 *       Revision:  none
 *       Compiler:  gcc
 *         Author:  YOUR NAME (), 
 *        Company:  
 * ************************************************************************/


#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
void cath_child(int num)
{
	pid_t pid;
	while(1)
	{
		pid = waitpid(-1,NULL,WNOHANG);
		if(pid <= 0)
		{
			break;
		}
		else if(pid > 0)
		{
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
	if(lfd <0)
		perror("");
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
		int cfd = accept(lfd,(struct sockaddr*)&cliaddr,&len);
		if(cfd < 0)
			perror("");
		char ip[16]="";
		printf("client ip=%s port=%d\n",
				inet_ntop(AF_INET,&cliaddr.sin_addr.s_addr,ip,16),
				ntohs(cliaddr.sin_port));
		pid_t pid;
		pid = fork();
		if(pid ==0 )//子进程
		{
			close(lfd);
			while(1)
			{
				sleep(1);
				char buf[256]="";
				int n = read(cfd,buf,sizeof(buf));
				if(0==n)
				{
					printf("client close\n");
					break;
				}
				printf("[*********%s*********]\n",buf);
				write(cfd,buf,n);
			
			}
			//close(cfd);
			//exit(0);
			break;
		}
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


	// 创建子进程
	 }
	
	// 收尾

	return 0;
}


