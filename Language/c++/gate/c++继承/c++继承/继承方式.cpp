#include<iostream>
using namespace std;
class Base1 {
public:
	int a;
protected:
	int b;
private:
	int c;
};
class Son1 : public Base1 {
public:
	void func() {
		cout << "a = " << a << " b = " << b << endl;
		//cout << "c = " << c << endl;
	}
};
int main1() {
	Son1 s;
	s.func();
	//类外无法访问
	//cout << s.b << endl;
	//cout << s.c << endl;
	return 0;
}