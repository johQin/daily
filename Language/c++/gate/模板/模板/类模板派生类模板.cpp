#include<iostream>
using namespace std;

template<class T1, class T2>
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

template<class T1,class T2,class T3>
class Sub:public Base<T1, T2> {
public:
	T3 c;
public:
	Sub(T1 a, T2 b, T3 c) :Base<T1, T2>(a, b) {
		this->c = c;
	}
};
int main() {
	Sub<int, char, int> s(10, 'a', 100);
	s.toString();
	cout << s.c << endl;
	return 0;
}
//基本类型：支持
//
//上行转换：支持 安全
//
//下行转换：支持 （不安全）
//
//不相关类型转换：不支持