#include<iostream>
using namespace std;

template<class T1,class T2>
class Base {
private:
	T1 a;
	T2 b;
public:
	Base() {};
	Base(T1 a, T2 b) {
		this->a = a;
		this->b = b;
	}
	void toString() {
		cout << "a = " << a << " b = " << b << endl;
	}
};

// 普通类继承类模板，模板类型必须具体化，否则无法派生
class Sub:public Base<int,char>{
public:
	int c;
public:
	Sub(int a, char b, int c) :Base(a, b) {
		this->c = c;
	}
};
int main10() {
	Sub s(10, 'a', 100);
	s.toString();
	cout << s.c << endl;
	return 0;
}