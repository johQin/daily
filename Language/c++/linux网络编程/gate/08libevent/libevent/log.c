
#include "log.h"

static int  msgopt, wanopt;
static char msgdatefmt[100], wandatefmt[100], ident_name[100];
static struct timeval be_stime;
static FILE *msgfile = NULL, *wanfile = NULL;
/* ************************************************************************************ */
/* ************************************************************************************ */
/* ************************************************************************************ */
//日志文件初始化,也可以通过msgLogOpen进行初始化
int msgInit(char *pName)
{
    if (msgLogOpen(pName, LOG_MESSAGE_FILE, LOG_POSTFIX_MESS,LOG_WARNING_FILE, LOG_POSTFIX_WARN) == 0)
    {
        msgLogFormat(LOG_PROCNAME|LOG_PID, LOG_MESSAGE_DFMT, LOG_PROCNAME|LOG_PID, LOG_WARNING_DFMT);
    }
    else                                                                                                
    {
        printf("can not create log!\n");
        return -1;
    }
    return 0;
}
/* ************************************************************************************ */
int msgLogOpen(char *ident, char *mpre, char *mdate, char *wpre, char *wdate) /* 打开日志 */
{
    time_t now_time;
    char openfilename[200], timestring[100];

    now_time = time(NULL);
    if ((!msgfile) && (*mpre))
    {
        strcpy(openfilename, mpre);
        if (*mdate)
        {
            strftime(timestring, sizeof(timestring), mdate, localtime(&now_time));
            strcat(openfilename, ".");
            strcat(openfilename, timestring);
        }
        if ((msgfile = fopen(openfilename, "a+b")) == NULL)
        { /* 如果没有应该把目录建上 */
            printf("openfilename=%s\n", openfilename);
            return -1;
        }
        setlinebuf(msgfile);
    }
    if ((!wanfile) && (*wpre))
    {
        strcpy(openfilename, wpre);
        if (*wdate)
        {
            strftime(timestring, sizeof(timestring), wdate, localtime(&now_time));
            strcat(openfilename, ".");
            strcat(openfilename, timestring);
        }
        if ((wanfile = fopen(openfilename, "a+b")) == NULL)
        {
            return -1;
        }
        setlinebuf(wanfile);
    }
    if ((msgfile) && (wanfile))
    {
        if (*ident)
        {
            strcpy(ident_name, ident);
        } else {
            ident_name[0] = '\0';
        }
        msgopt = LOG_PROCNAME|LOG_PID;          /* 设置默认信息输出信息选项              */
        wanopt = LOG_PROCNAME|LOG_PID;          /* 设置默认告警输出信息选项              */
        strcpy(msgdatefmt, "%m-%d %H:%M:%S");   /* 默认信息输出时间格式 MM-DD HH24:MI:SS */
        strcpy(wandatefmt, "%m-%d %H:%M:%S");   /* 默认告警输出时间格式 MM-DD HH24:MI:SS */

        msglog(MSG_INFO,"File is msgfile=[%d],wanfile=[%d].",fileno(msgfile),fileno(wanfile));
        return 0;
    } else {
        return -1;
    }
}
/* ************************************************************************************ */
/* 自定义日志输出函数系列,可以按普通信息及告警信息分类输出程序日志                      */
int msglog(int mtype, char *outfmt, ...)
{
    time_t now_time;
    va_list ap;//变参的列表
    char logprefix[1024], tmpstring[1024];

    time(&now_time);
    if (mtype & MSG_INFO)
    { /*strftime会将localtime(&now_time)按照msgdatefmt格式,输出到logprefix.*/
        strftime(logprefix, sizeof(logprefix), msgdatefmt, localtime(&now_time));
        strcat(logprefix, " ");
        /*static int  msgopt,wanopt;*/
        if (msgopt&LOG_PROCNAME)
        {
            strcat(logprefix, ident_name);
            strcat(logprefix, " ");
        }
        if (msgopt&LOG_PID)
        {
            sprintf(tmpstring, "[%6d]", getpid());
            strcat(logprefix, tmpstring);
        }
        fprintf(msgfile, "%s: ", logprefix);
        va_start(ap, outfmt);
        vfprintf(msgfile, outfmt, ap);
        va_end(ap);
        fprintf(msgfile, "\n");
    }
    if (mtype & MSG_WARN)
    {
        strftime(logprefix, sizeof(logprefix), wandatefmt, localtime(&now_time));
        strcat(logprefix, " ");
        /*#define LOG_PROCNAME      0x00000001*/              /* msglog 输出日志时打印程序名        */
        if (wanopt & LOG_PROCNAME)
        {
            strcat(logprefix, ident_name);
            strcat(logprefix, " ");
        }
        if (wanopt & LOG_PID)
        {
            sprintf(tmpstring, "[%6d]", getpid());
            strcat(logprefix, tmpstring);
        }
        fprintf(wanfile, "%s: ", logprefix);
        va_start(ap, outfmt);
        vfprintf(wanfile, outfmt, ap);
        va_end(ap);
        fprintf(wanfile, "\n");
        if (wanopt & LOG_PERROR)
        {
            fprintf(stderr, "%s: ", logprefix);
            va_start(ap, outfmt);
            vfprintf(stderr, outfmt, ap);
            va_end(ap);
            fprintf(stderr, "\n");
        }
    }

    return 0;
}
/* ************************************************************************************ */
int msgLogFormat(int mopt, char *mdfmt, int wopt, char *wdfmt)   /* 设置日志格式及选项  */
{
    if (mopt >= 0)
    {
        msgopt = mopt;
    } else {
        msgopt = msgopt & mopt;
    }
    if (wopt >= 0)
    {
        wanopt = wopt;
    } else {
        wanopt = wanopt & wopt;
    }

    if (*mdfmt) strcpy(msgdatefmt, mdfmt);
    if (*wdfmt) strcpy(wandatefmt, wdfmt);

    return 0;
}
/* ************************************************************************************ */
int msgLogClose(void)                                           /* 关闭日志文件         */
{
    if (msgfile) fclose(msgfile);
    if (wanfile) fclose(wanfile);

    return 0;
}
/* ************************************************************************************ */
long begusec_process(void)                      /* 设置开始时间 0=ok                    */
{
    gettimeofday(&be_stime,NULL);

    return 0;
}
/* ************************************************************************************ */
long getusec_process(void)                      /* 返回usecond 从 begusec_process历时   */
{
    struct timeval ed_stime;

    gettimeofday(&ed_stime,NULL);

    return ((ed_stime.tv_sec-be_stime.tv_sec)*1000000+ed_stime.tv_usec-be_stime.tv_usec);
}
/* ************************************************************************************ */
/* ************************************************************************************ */
/* ************************************************************************************ */
/* ************************************************************************************ */
