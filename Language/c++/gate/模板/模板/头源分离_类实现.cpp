#include "ͷԴ����.h"
template<class T1, class T2>
Data<T1, T2>::Data() {
	cout << "�޲ι���" << endl;
}

template<class T1, class T2>
Data<T1, T2>::Data(T1 a, T2 b) {
	this->a = a;
	this->b = b;
}

template<class T1, class T2>
void Data<T1, T2>::toString() {
	cout << "a = " << a << " b = " << b << endl;
}