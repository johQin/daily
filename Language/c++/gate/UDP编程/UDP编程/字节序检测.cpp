#include<stdio.h>

typedef union Data {
	unsigned short a;
	char b[2];//b[0]��b[1]�ֱ������a�������ֽ�
} data;
int main1() {
	data tmp;
	tmp.a = 0x6141;// 16��������61����ʮ���Ƶ�97��a)����16����41��ʮ���Ƶ�65��A��
	if (tmp.b[0] == 0x61) {
		printf("%c�����", tmp.b[0]);//��ӡ��a��˵����0x61�����Ǵ�ˣ������ǵ��Ķ�ϰ��һ��
	}
	else {
		printf("%c��С��", tmp.b[0]);//��ӡ��A��˵����0x41������С�ˣ������ǵ��Ķ�ϰ�߲�һ��
	}

	printf("%c", 34);
	return 0;
}