#include<wrap.h>
#include<sys/epoll.h>
// 本文件中，首字母大写的函数都在wrap.c中
int main(int argc, char *argv[]){
    //创建套接字，并绑定
    int lfd = tcp4bind(7777, NULL);// 在wrap.c中实现
    //监听
    listen(lfd, 128);
    // 创建树
    int epfd = epoll_create(1);
    // 将lfd上树
    struct epoll_event ev;
    ev.events = EPOLL_IN;//读事件
    ev.data.fd = lfd;
    epoll_ctl(epfd,EPOLL_CTL_ADD, lfd, &ev);
    //循环监听树
    struct epoll_event evs[1024];//存放变化的fds
    while(1){
        int n = epoll_wait(epfd, evs, 1024, -1)// -1将无限期的阻塞在这里
        if(n<0){
            perror("");
            exit(-1);
        }else if(>=0){
            for(int i; i<n; i++){
                int fd = evs[i].data.fd;
                //如果是lfd变化，并且是读事件变化
                if(fd == lfd && evs[i].events&EPOLLIN){//位与运算

                    struct  sockaddr_in cliaddr;
                    socklen_t len = sizeof(cliaddr);
                    char ip[16] = ""
                    int cfd = Accept(lfd,(struct sockaddr*) &cliaddr,&len);
                    printf("client ip = %s port =%n\n",
                        inet_ntop(AF_INET, &cliaddr.sin_addr, ip, 16),
                        ntohs(cliaddr.sin_port);
                    )
                    //将cfd上树
                    ev.data.fd = cfd;
                    ev.events = EPOLLIN;
                    epoll_ctl(epfd, EPOLL_CTL_ADD, cfd, &ev);

                }else if(evs[i].events&EPOLLIN){//cfd变化，并且是读事件
                
                    char buf[1500] = "";
                    int count = Read(fd, buf, sizeof(buf));
                    if(count<0){
                        printf("error or client close\n");
                        close(fd);
                        epoll_ctl(epfd, EPOLL_CTL_DEL, fd, &evs[i]);
                    }else {
                        printf("%s\n",buf);
                        Write(fd,buf,count);
                    }

                }
            }
        }
    }
    
}