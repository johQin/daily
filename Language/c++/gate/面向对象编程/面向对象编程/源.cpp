#include<iostream>
using namespace std;
class Data{
public:
	Data() {
		cout << "�޲ι���" << endl;
	}
	~Data() {
		cout << "����" << endl;
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
	// ��->û����֮ǰ������������func�ġ�
	//sp.p->func();
	// ->����֮��
	sp->func();

	// *���غ�
	(*sp).func();
	return 0;
}