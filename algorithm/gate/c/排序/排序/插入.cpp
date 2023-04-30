#include<stdio.h>
#include<iostream>
using namespace std;
void main()
{
	char a[] = "abc"; //栈
		char b[] = "abc"; //栈
		char* c = "abc"; //abc在常量区，c在栈上。
		char* d = "abc"; //编译器可能会将它与c所指向的"abc"优化成一个地方。
		const char e[] = "abc"; //栈
		const char f[] = "abc"; //栈

		cout << a << " " << b << " " << c << " " << d << " " << e << " " << f << endl;
	cout << (a == b ? 1 : 0) << endl << (c == d ? 1 : 0) << endl << (e == f ? 1 : 0) << endl;
}