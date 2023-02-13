#include<iostream>
using namespace std;


int main3() {
	int a = 10;

	//将const修饰的指针或引用转换为 非const
	const int* p1 = &a;
	int* p2 = const_cast<int*>(p1);

	const int& ob=20;
	int& ob1 = const_cast<int&>(ob);


	int b = 20;
	
	//将非const修饰的指针或引用 转换成 const （支持）
	int* p3 = &b;
	const int* p4 = const_cast<const int*>(p3);

	int c = 30;
	int& ob2 = c;
	const int& p5 = const_cast<const int&>(ob2);


	return 0;
}