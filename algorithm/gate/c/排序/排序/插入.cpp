#include<stdio.h>
#include<iostream>
using namespace std;
void main()
{
	char a[] = "abc"; //ջ
		char b[] = "abc"; //ջ
		char* c = "abc"; //abc�ڳ�������c��ջ�ϡ�
		char* d = "abc"; //���������ܻὫ����c��ָ���"abc"�Ż���һ���ط���
		const char e[] = "abc"; //ջ
		const char f[] = "abc"; //ջ

		cout << a << " " << b << " " << c << " " << d << " " << e << " " << f << endl;
	cout << (a == b ? 1 : 0) << endl << (c == d ? 1 : 0) << endl << (e == f ? 1 : 0) << endl;
}