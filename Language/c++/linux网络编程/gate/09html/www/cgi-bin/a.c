#include <stdio.h>
#include <stdlib.h>
int main(void)
{
	printf("content-type:text/html\n\n");
	char *data = NULL;
	data = getenv("QUERY_STRING");
	printf("%s\n",data);

	printf("<html>\n<TITLE>CGI1:CGI hello!</TITLE>\n");
	printf("<center><H1>hello, this is frist CGI demo!</H1></center>\n</html>");
	return 0;
}