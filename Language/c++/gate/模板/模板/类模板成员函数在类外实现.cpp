#include<iostream>
using namespace std;

template <class T1, class T2>
class Data {
private:
	T1 a;
	T2 b;
public:
	Data() {}
	Data(T1 a, T2 b);
	void toString();
};

// 由于模板只作用于当前类，所以成员函数在类外实现，必须重新声明模板。
// 类模板的构造器在类外实现
template <class T1, class T2>
Data<T1, T2>::Data(T1 a, T2 b)
{
	this->a = a;
	this->b = b;
}

// 类模板的成员函数在类外实现
template <class T1, class T2>
// 不管成员函数是否携参，在类外实现都必须附上类的作用域
// 而类模板的作用域书写应该为Data<T1,T2>
// 既然此处用到了T1，T2，那么就必须在语句前面加上template
void Data<T1,T2>::toString()
{
	cout << "a = " << a << " b = " << b << endl;
}

int main4() {
	Data<int, int> ob(10, 20);
	ob.toString();
	return 0;
}