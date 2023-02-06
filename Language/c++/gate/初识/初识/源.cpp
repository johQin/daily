#include<iostream>

using namespace std;
int main() {
	int a1 = 10;
	int b1 = 3;
	// 两个整数相除，结果依然是一个整数，会直接去掉小数部分（不会四舍五入）
	cout << a1 / b1 << endl;

	double a2 = 3.14;
	double b2 = 0.1;
	// 两个小数不能进行取余运算
	//cout << a2 % b2 < endl;
	if(a1>b1) cout << a1 / b1 << endl;
}
