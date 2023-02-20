#include <iostream>
using namespace std;
template <class T, class F>
void func(T t, F f)
{
	static int a = 0; // 如果模板只实例化一次，每个a的地址是一样的
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