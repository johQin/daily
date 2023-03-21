#ifndef LOG_H
#define LOG_H
/* ************************************************************************************ */

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/types.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#define LOG_PROCNAME      0x00000001              /* msglog �����־ʱ��ӡ������        */
#define LOG_PID           0x00000010              /* msglog �����־ʱ��ӡ���� PID      */
#define LOG_PERROR        0x00000100              /* msglog �Ƿ�Ѹ澯���������stderr  */
#define NLO_PROCNAME      0x11111110              /* msglog �����������                */
#define NLO_PID           0x11111101              /* msglog ��������� PID              */
#define NLO_PERROR        0x11111011              /* msglog ������澯��stderr          */

#define MSG_INFO          0x00000001              /* msglog ������澯��־�ļ���        */
#define MSG_WARN          0x00000010              /* msglog �������ͨ��־�ļ���        */
#define MSG_BOTH          MSG_INFO|MSG_WARN       /* msglog �������ͨ�͸澯��־�ļ���  */

#define LOG_MESSAGE_FILE  "/home/itheima/log/tcpsvr"           /* ϵͳ����������־��Ϣ�ļ�           */
#define LOG_MESSAGE_DFMT  "%m-%d %H:%M:%S"        /* ��־��Ϣʱ���ʽ�ִ�               */
#define LOG_POSTFIX_MESS  "%y%m"                  /* ����������־��Ϣ�ļ���׺           */
#define LOG_WARNING_FILE  "/home/itheima/log/log.sys_warn"   /* ϵͳ�������и澯��־�ļ�           */
#define LOG_WARNING_DFMT  "%m-%d %H:%M:%S"        /* �澯��Ϣʱ���ʽ�ִ�               */
#define LOG_POSTFIX_WARN  ""                      /* �������и澯��־�ļ���׺           */

/* ************************************************************************************ */
int msglog(int mtype, char *outfmt, ...);//д��־����
int msgLogFormat(int mopt, char *mdfmt, int wopt, char *wdfmt);//����־��ʽ�� 
int msgLogOpen(char *ident, char *mpre, char *mdate, char *wpre, char *wdate);//����־�ļ�
int msgLogClose(void);//�ر���־�ļ�

long begusec_process(void);                      /* ���ÿ�ʼʱ�� 0=ok                   */
long getusec_process(void);                      /* ����usecond �� begusec_process��ʱ  */

int msgInit(char *pName);
#endif
/* ************************************************************************************ */
/* ************************************************************************************ */
/* ************************************************************************************ */
/* ************************************************************************************ */

