#include<iostream>
#include<string.h>
using namespace std;
class Base4 {
public:
	int a;
	string b;
	Base4(int a) {
		this->a = a;
		b = "����b";
	}
	void func() {
		cout << "����func��a = " << a << endl;
	}
	void func01() {
		cout << "����func01��b = " << b << endl;
	}
};
class Son4 : public Base4 {
public:
	int a;
	Son4(int x, int y) :Base4(x) {
		a = y;
	}
	void func() {
		cout << "����func��a = " << a << endl;
	}
};
int main4() {
	Son4 s(10,20);

	// ����Ĭ�����Ȳ�����������û�ж�Ӧ�ĳ�Ա
	// �������ֱ�ӷ���
	// ���û������������ڸ����в���
	cout<<s.a<<endl; // 20
	cout << s.b << endl; //����b
	s.func();// ����func
	s.func01();// ����func01

	// ����ʱ�������������֣�������ȷʹ�ø����Ӧ�ĳ�Ա
	cout<<s.Base4::a<<endl;
	s.Base4::func();
	return 0;
}