#include<iostream>
#include<string.h>
using namespace std;

class Data{
private:
	int a;
public:
	//friend ostream& operator<<(ostream& out, Data ob);
public:
	Data() {};
	Data(int a) {
		this->a = a;
	}
};
//ostream& operator<<(ostream& out, Data ob) {
//	cout << ob.a << endl;
//	return out;
//}
template<typename T>
void myPrintAll(T a) {
	cout  << a  << endl;
}

int main() {
	Data d(10);
	myPrintAll(d);

	return 0;
}
