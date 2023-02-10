#include<iostream>
#include<string.h>
using namespace std;
class Base4 {
public:
	int a;
	string b;
	Base4(int a) {
		this->a = a;
		b = "父类b";
	}
	void func() {
		cout << "父类func：a = " << a << endl;
	}
	void func01() {
		cout << "父类func01：b = " << b << endl;
	}
};
class Son4 : public Base4 {
public:
	int a;
	Son4(int x, int y) :Base4(x) {
		a = y;
	}
	void func() {
		cout << "子类func：a = " << a << endl;
	}
};
int main4() {
	Son4 s(10,20);

	// 子类默认优先查找自身类有没有对应的成员
	// 如果有则直接返回
	// 如果没有则继续向上在父类中查找
	cout<<s.a<<endl; // 20
	cout << s.b << endl; //父类b
	s.func();// 子类func
	s.func01();// 父类func01

	// 重名时，加作用域区分，即可正确使用父类对应的成员
	cout<<s.Base4::a<<endl;
	s.Base4::func();
	return 0;
}