#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <termios.h>

#define GREEN 	"\e[32m"   //shell打印显示绿色
#define RED     "\e[31m"   //shell打印显示红色
#define PRINT(X,Y) {	write(1,Y,5);  \
					printf(X);   \
					fflush(stdout); \
				    write(1,"\e[0m",4); \
				 }

static int sockfd;
static struct sockaddr_in dest_addr;

void sig_dispose(int sig)
{
	if(SIGINT == sig){
		close(sockfd);
		puts("\nclose!");
		system("stty sane");//回显
		exit(0);
	}
}
//上传
void tftp_upload(char *argv)
{
	int fd,read_len;
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
	
	//构造上传请求,argv为文件名
	int len = sprintf(cmd_buf, "%c%c%s%c%s%c", 0, 2, argv, 0, "octet", 0);	//发送读数据包请求
	sendto(sockfd, cmd_buf, len, 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
	
	fd = open(argv, O_RDONLY);
	if(fd < 0 ){
		perror("open error");
		close(sockfd);
		exit(-1);
	}
	
	do{
		//接收服务器发送的内容
		len = recvfrom(sockfd, recv_buf, sizeof(recv_buf), 0, (struct sockaddr*)&client_addr, &cliaddr_len);
		
		cmd = recv_buf[1];
		if( cmd == 4 )	//是否为ACK
		{
			p_num = ntohs(*(unsigned short*)(recv_buf+2));
			read_len = read(fd, recv_buf+4, 512);
			printf("recv:%d\n", p_num);//十进制方式打印包编号		
			recv_buf[1] = 3;//构建数据包
			(*(unsigned short *)(recv_buf+2)) = htons(p_num + 1);
			//printf("%s\n", recv_buf+3);
			sendto(sockfd, recv_buf, read_len+4, 0, (struct sockaddr*)&client_addr, sizeof(client_addr));
		}
		else if( cmd == 5 ) //是否为错误应答
		{
			close(sockfd);
			close(fd);
			printf("error:%s\n", recv_buf+4);
			exit(-1);
		}		
	}while(read_len == 512); //读取的数据小于512则认为结束
	len = recvfrom(sockfd, recv_buf, sizeof(recv_buf), 0, (struct sockaddr*)&client_addr, &cliaddr_len);      //接收最后一个ACK确认包
	close(fd);
	PRINT("Download File is Successful\n", RED);
	return;}


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
void help_fun(int argc, char *argv[])
{
	printf("1  down\n");
	printf("2  upload\n");
	printf("3  exit\n");
	return;
}

char mygetch() 
{
    struct termios oldt, newt;
    char ch;
    tcgetattr( STDIN_FILENO, &oldt );
    newt = oldt;
    newt.c_lflag &= ~( ICANON | ECHO );
    tcsetattr( STDIN_FILENO, TCSANOW, &newt );
    ch = getchar();
    tcsetattr( STDIN_FILENO, TCSANOW, &oldt );
    return ch;
}

int main(int argc, char *argv[])
{
	char cmd_line[100];
	signal(SIGINT,sig_dispose);
	
	bzero(&dest_addr, sizeof(dest_addr));
	sockfd = socket(AF_INET, SOCK_DGRAM, 0);
	if(sockfd < 0){
		perror("socket error");
		exit(-1);
	}
	while(1){
		int cmd;
		help_fun(argc,argv);
		PRINT("send>", GREEN);
		cmd = mygetch();
		if(cmd == '1'){
			puts("input file name:");
			fgets(cmd_line,sizeof(cmd_line),stdin);
			*(strchr(cmd_line,'\n')) = '\0';
			tftp_down(cmd_line);
		}
		else if(cmd == '2')	{
			puts("input file name:");
			fgets(cmd_line,sizeof(cmd_line),stdin);
			*(strchr(cmd_line,'\n')) = '\0';
			tftp_upload(cmd_line);
		}			
		else if(cmd == '3')	{
			close(sockfd);			
			system("stty sane");//回显
			exit(0);
		}
	}
	
	close(sockfd);
	return 0;
}
