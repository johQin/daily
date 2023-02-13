#include<iostream>
using namespace std;

template <class T1, class T2>
class Data {
	template<class T3, class T4>
	friend void myPrint(Data<T3, T4>& ob);
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

template<class T3, class T4> void myPrint(Data<T3, T4>& ob) {
	cout << ob.a << " " << ob.b << endl;
}

int main5() {
	Data<int, int> ob(10, 20);
	myPrint(ob);
	return 0;
}