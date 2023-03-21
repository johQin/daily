#include "log.h"


int main(int argc,char *argv[])
{
    char *pName = argv[0];
    pName +=2;
    printf("pName is %s\n",pName);
    msgInit("log_demo");
    msglog(MSG_INFO,"begin run program....");
    sleep(2);
    msglog(MSG_BOTH,"begin to game over...%ld",time(NULL));
    msgLogClose();
    return 0;
}

