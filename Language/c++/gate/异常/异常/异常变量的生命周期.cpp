#include<iostream>
using namespace std;
class MyException {
public:
	MyException() {
		cout << "异常变量无参构造" << endl;
	}
	MyException(const MyException& e) {
		cout << "拷贝构造" << endl;
	}
	~MyException() {
		cout << "异常变量析构" << endl;
	}
};
//1.普通变量接手异常
void test01(){
	try {
		throw MyException();//构造
	}
	catch (MyException e) {// 发生拷贝构造，效率略低
		cout << "普通变量接手异常" << endl;
	}
}
//2.指针对象接手异常
void test02() {
	try {
		throw new MyException;//构造
	}
	catch (MyException *e) {
		cout << "指针接手异常" << endl;
		delete e;//析构，需要人为手动释放。
	}
}
//3. 引用接手异常,析构无需手动
void test03() {
	try {
		throw MyException();//构造
	}
	catch (MyException& e) {
		cout << "引用接手异常" << endl;
	}
}
int main2() {
	test01();
	//异常变量无参构造
	//拷贝构造
	//普通变量接手异常
	//异常变量析构
	//异常变量析构
	cout<<"-----------" << endl;
	test02();
	//异常变量无参构造
	//指针接手异常
	//异常变量析构
	cout << "-----------" << endl;
	test03();
	//异常变量无参构造
	//引用接手异常
	//异常变量析构
	return 0;
}