#include<iostream>
#include<string>
#include<memory>

using namespace std;
class T1{};
class T2{};
void func(shared_ptr<T1>, shared_ptr<T2>){}
int main2() {
	// 方式1：匿名对象直接构造智能指针（不安全）
	shared_ptr<T1> ptr1(new T1());
	shared_ptr<T2> ptr2(new T2());
	func(ptr1, ptr2);

	// main函数执行步骤：
	//1、分配内存给T1
	//2、分配内存给T2
	//3、构造T1对象
	//4、构造T2对象
	//5、构造T1的智能指针对象
	//6、构造T2的智能指针对象
	//7、调用func

	// 如果程序在执行第3步失败，那么在第1,2步，分配给T1和T2的内存将会造成泄漏。
	// 解决这个问题很简单，不要在shared_ptr构造函数中使用匿名对象。
	// 
	// 方式2：优先选用make_shared（c++11）/make_unique（c++14）而非直接使用new。（安全）
	// 简单说来，相比于直接使用new表达式，make系列函数有三个优点：消除了重复代码、改进了异常安全性和生成的目标代码尺寸更小速度更快

	shared_ptr<T1> ptr3 = make_shared<T1>();
	shared_ptr<T2> ptr4 = make_shared<T2>();

	func(ptr3, ptr4);
	return 0;


}