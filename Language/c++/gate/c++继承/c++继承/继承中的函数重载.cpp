#include<iostream>
#include<string.h>
using namespace std;
class Base5 {
public:
	void func() {
		cout << "父类func()" << endl;
	}
	void func(int a) {
		cout << "父类func(int a)" << endl;
	}
	void func(float a) {
		cout << "父类func(float a)" << endl;
	}
};
class Son5 : public Base5 {
public:
	void func() {
		cout << "子类func()" << endl;
	}
};
int main5() {
	Son5 s;
	
	s.func();

	//屏蔽所有父类中同名函数，重载效果消失
	//s.func(1);
	//s.func(1.05f);

	//父类中的重载函数依然被继承，但是需要加作用域
	s.Base5::func(1);
	s.Base5::func(1.05f);
	return 0;
}