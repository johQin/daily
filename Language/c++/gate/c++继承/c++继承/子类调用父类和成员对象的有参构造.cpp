#include<iostream>
using namespace std;
class Base3 {
public:
	int a;
public:
	Base3() {
		cout << "父类构造" << endl;
	}
	Base3(int a) {
		this->a = a;
		cout << "父类有参构造" << endl;
	}

};
class Member3 {
public:
	int b;
public:
	Member3() {
		cout << "子类成员构造" << endl;
	}
	Member3(int b) {
		this->b = b;
		cout << "子类成员有参构造" << endl;
	}
};
class Son3 : public Base3 {

public:
	int c;
	Member3 m;
public:
	Son3() {
		cout << "子类构造" << endl;
	}
	Son3(int a, int b, int c) :Base3(a),m(b),c(c){
		cout << "子类有参构造" << endl;
	}
};
int main3() {
	Son3 s(1,2,3);
	return 0;

}