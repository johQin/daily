#include<iostream>
using namespace std;
class Data{
public:
	Data() {
		cout << "无参构造" << endl;
	}
	~Data() {
		cout << "析构" << endl;
	}
	void func() {
		cout << "func" << endl;
	}
};
class SmartPointer {
public:
	Data* p;
public:
	SmartPointer(Data* p) {
		this->p = p;
	}
	~SmartPointer() {
		delete p;
	}
	Data* operator->() {
		return p;
	}
	Data& operator*() {
		return *p;
	}
};
int main() {
	/*
	Data* p = new Data;
	p->func();
	delete p;
	*/

	SmartPointer sp(new Data);
	// 在->没重载之前，是这样调用func的。
	//sp.p->func();
	// ->重载之后
	sp->func();

	// *重载后
	(*sp).func();
	return 0;
}