#include<iostream>
using namespace std;
class Base3 {
public:
	int a;
public:
	Base3() {
		cout << "���๹��" << endl;
	}
	Base3(int a) {
		this->a = a;
		cout << "�����вι���" << endl;
	}

};
class Member3 {
public:
	int b;
public:
	Member3() {
		cout << "�����Ա����" << endl;
	}
	Member3(int b) {
		this->b = b;
		cout << "�����Ա�вι���" << endl;
	}
};
class Son3 : public Base3 {

public:
	int c;
	Member3 m;
public:
	Son3() {
		cout << "���๹��" << endl;
	}
	Son3(int a, int b, int c) :Base3(a),m(b),c(c){
		cout << "�����вι���" << endl;
	}
};
int main3() {
	Son3 s(1,2,3);
	return 0;

}