#include <iostream>
using namespace std;
template <class T, class F>
void func(T t, F f)
{
	static int a = 0; // ���ģ��ֻʵ����һ�Σ�ÿ��a�ĵ�ַ��һ����
	cout << &a << endl;
}

int main1()
{
	func(1, 1);
	func(1, 2);
	func(1, 1.1);
	func('1', 2);
	return 0;
}
//00007FF71901C170
//00007FF71901C170
//00007FF71901C174
//00007FF71901C178