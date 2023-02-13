#include<iostream>
#include<string.h>
using namespace std;

// 当调用模板函数的时候，如果类型自动推导出模板类型为Data，那么就会调用具体化函数模板
template<typename T>
void myPrintAll(T a) {
	cout << a << endl;
}
class Data{
private:
	int a;
public:
	friend void myPrintAll<Data>(Data d);
public:
	Data() {};
	Data(int a) {
		this->a = a;
	}
};


template<>
void myPrintAll<Data>(Data d) {
	cout << d.a << endl;
}
int main1() {
	Data d(10);
	myPrintAll(d);

	return 0;
}
