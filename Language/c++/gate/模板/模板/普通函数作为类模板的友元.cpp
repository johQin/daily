#include<iostream>
using namespace std;

template <class T1, class T2>
class Data {
	friend void myPrint(Data<int, char>& ob);
private:
	T1 a;
	T2 b;
public:
	Data() {}
	Data(T1 a, T2 b) {
		this->a = a;
		this->b = b;
	}
};

void myPrint(Data<int, char>& ob) {
	cout << ob.a << " " << ob.b << endl;
}

int main6() {
	Data<int, char> ob(10, 'b');
	myPrint(ob);
	return 0;
}