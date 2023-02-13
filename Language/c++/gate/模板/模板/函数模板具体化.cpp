#include<iostream>
#include<string.h>
using namespace std;

// ������ģ�庯����ʱ����������Զ��Ƶ���ģ������ΪData����ô�ͻ���þ��廯����ģ��
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
