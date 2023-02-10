#include<iostream>
using namespace std;
class Base2 {
public:
	Base2() {
		cout << "父类构造" << endl;
	}
	~Base2() {
		cout << "父类析构" << endl;
	}
};
class Member2 {
public:
	Member2() {
		cout << "子类成员构造" << endl;
	}
	~Member2() {
		cout << "子类成员析构" << endl;
	}
};
class Son2 : public Base2 {
public:
	Member2 m;
public:
	Son2() {
		cout << "子类构造" << endl;
	}
	~Son2() {
		cout << "子类析构" << endl;
	}
};
int main2() {
	Son2 s;

	return 0;
	//父类构造
	//子类成员构造
	//子类构造

	//子类析构
	//子类成员析构
	//父类析构
}