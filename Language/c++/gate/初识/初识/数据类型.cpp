#include<iostream>
using namespace std;
int main3() {

	// 1.整型数据
	short num1 = 10;
	int num2 = 10;
	//unsigned num2 = 10;
	//unsigned int num2 = -10; //会输出它的补码
	long num3 = 10;
	long long num4 = 10;
	cout << "num1 = " << num1 << endl;
	cout << "num2 = " << num2 << endl;
	cout << "num3 = " << num3 << endl;
	cout << "num4 = " << num4 << endl;

	//2.利用sizeof计算数据类型或变量所占的空间大小
	//sizeof(type or var)，单位：字节
	cout << "short所占的空间：" << sizeof(short) << endl;
	cout << "int所占的空间：" << sizeof(int) << endl;
	cout << "short num1所占的空间：" << sizeof(num1) << endl;

	//3.实型
	//默认情况下，输出一个小数，会显示出6位有效数字
	float f1 = 3.14f;
	double d1 = 3.14;
	cout << "f1= " << f1 << endl;
	cout << "d1= " << d1 << endl;
	cout << "float所占空间" << sizeof(float) << endl;
	cout << "double所占空间" << sizeof(double) << endl;
	float f2 = 3e2; //3*10^2
	float f3 = 3e-2;
	cout << "f2= " << f2 << endl;
	cout << "f3= " << f3 << endl;

	//4.字符型
	//只占一个字节，字符型变量并不是把字符本身放到内存中存储，而是将对应的ASCII编码放入到存储单元
	char ch1 = 'a';
	cout << "ch1= " << ch1 << endl;
	cout << "char所占空间 " << sizeof(char) << endl;
	// 字符型变量的值只能包含一个字符，且只能用单引号，赋值不能用双引号，"b"
	cout << "ch1='a'对应的ASCII码值" << (int)ch1 << endl;

	//5.字符串
	// 要用双引号
	// a.c语言风格字符数组
	char ch2[] = "abcdefg";
	// b.c++风格的字符串要#include<string>,有些集成环境不需要，但是希望还是要写上。
	//#include <bits/stdc++.h>无敌！
	string ch3 = "hijklmn";
	cout << "ch2 =" << ch2 << endl;

	//6.bool
	bool flag1 = true;
	bool flag2 = false;
	cout << "flag1 =" << flag1 << endl;// 打印1
	cout << "flag2=" << flag2 << endl;
	cout << "flag2占用内存" << sizeof(flag2) << endl; // 1

	//7.输入和输出
	int a = 0;
	cout << "请给int类型a赋值" << endl;
	cin >> a;
	cout << "int类型a="<< a << endl;
	return 0;
}