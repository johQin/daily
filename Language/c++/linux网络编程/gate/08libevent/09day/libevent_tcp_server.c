#include "wrap.h"
#include <event.h>
#include <stdio.h>
#define PORT 8888
void cfdcb(evutil_socket_t fd, short events, void *arg)
{
	struct event_base  *base = (struct event_base  *)arg;
	char buf[1024]="";
	int n = Read(fd,buf,sizeof(buf));
	if(n <= 0)
	{
		//event_del();
		//close(fd);
		printf("client close\n");

	}
	else 
	{
		printf("%s\n",buf);
		write(fd,buf,n);

	}

}
void lfdcb(evutil_socket_t fd, short events, void *arg)
{

 	struct event_base  *base = ( struct event_base  *)arg;
 	struct sockaddr_in cliaddr;
 	socklen_t len = sizeof(cliaddr);
 	char ip[16]="";
 	int cfd = Accept(fd,(struct sockaddr*)&cliaddr,&len);
 	printf("ip=%s port=%d\n",inet_ntop(AF_INET,&cliaddr.sin_addr.s_addr,ip,16),
 			ntohs(cliaddr.sin_port));
 	struct event *new_node = event_new(base, cfd,EV_READ | EV_PERSIST, cfdcb, base);
	event_add(new_node, NULL);

}
int main(int argc, char const *argv[])
{
	
	int lfd = tcp4bind(PORT,NULL);
	Listen(lfd,128);
	//创建event_根
	 struct event_base  *base = event_base_new();
	 //初始ddna
	 struct event *new_node = event_new(base, lfd,EV_READ | EV_PERSIST, lfdcb, base);
	 event_add(new_node, NULL);
	 event_base_dispatch(base);

	 event_base_free(base);
	 event_free(new_node);

	return 0;
}