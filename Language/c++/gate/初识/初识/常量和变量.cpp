#include<iostream>
using namespace std;
//常量的定义方式
//1.符号常量（宏常量）
// 定义语句后面不能加“;”
#define Day 7
int main2() {
	cout << "你好，一周总共=" << Day << "天" << endl;
	
	// 2 const 修饰的常变量
	const int month = 12;
	cout << "one year have " << month << " month" << endl;

	return 0;
}