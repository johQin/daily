#include<iostream>
#include<string.h>
using namespace std;

// template 后面必须紧跟函数，并且只对当前函数有效，
template<typename T>
// 函数在调用时，会根据上下文推导出T的类型
// 然后会将函数内所有的T替换成int，重新对代码再编译一次，
void swapAll(T& a, T& b) {
	T tmp = a;
	a = b;
	b = tmp;
}
int main2() {
	int a = 10, b = 20;
	swapAll(a, b);
	cout << "a = " << a << " b = " << b << endl;

	string c = "hello", d = "world";
	swapAll(c, d);
	cout << "c = " << c << " d = " << d << endl;
	return 0;
}
