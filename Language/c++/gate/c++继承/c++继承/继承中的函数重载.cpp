#include<iostream>
#include<string.h>
using namespace std;
class Base5 {
public:
	void func() {
		cout << "����func()" << endl;
	}
	void func(int a) {
		cout << "����func(int a)" << endl;
	}
	void func(float a) {
		cout << "����func(float a)" << endl;
	}
};
class Son5 : public Base5 {
public:
	void func() {
		cout << "����func()" << endl;
	}
};
int main5() {
	Son5 s;
	
	s.func();

	//�������и�����ͬ������������Ч����ʧ
	//s.func(1);
	//s.func(1.05f);

	//�����е����غ�����Ȼ���̳У�������Ҫ��������
	s.Base5::func(1);
	s.Base5::func(1.05f);
	return 0;
}