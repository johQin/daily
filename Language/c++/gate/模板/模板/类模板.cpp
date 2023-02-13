#include<iostream>
using namespace std;

template <class T1,class T2>
class Data {
private:
	T1 a;
	T2 b;
public:
	Data(){}
	Data(T1 a, T2 b) {
		this->a = a;
		this->b = b;
	}
	void toString() {
		cout << "a = " << a << " b = " << b << endl;
	}
};
int main3() {
	Data<int,int> ob;
	ob.toString();
	Data<int, int> ob1(10, 20);
	ob1.toString();
	Data<int, char> ob2(10, 'b');
	ob2.toString();
	return 0;
}