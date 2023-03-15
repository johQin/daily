#include<stdio.h>
//#include<arpa/inet.h>
#include <windows.h>
#pragma comment(lib, "wsock32.lib")

int main() {
	int num4bytehost = 0x01020304;
	short num2bytehost = 0x0102;
	printf("%x\n", htonl(num4bytehost));
	printf("%x\n", htons(num2bytehost));
	return 0;
}