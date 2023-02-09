#include<iostream>
using namespace std;
class Print {
public:
	//重载函数调用符()
	void operator()(char* str) {
		cout << str << endl;
	}
};
int main10() {
	Print p;
	p("hello world");

	// Print()匿名对象，
	Print()("你好，世界");
	return 0;
}