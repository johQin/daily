#include<iostream>
using namespace std;
int main3() {

	// 1.��������
	short num1 = 10;
	int num2 = 10;
	//unsigned num2 = 10;
	//unsigned int num2 = -10; //��������Ĳ���
	long num3 = 10;
	long long num4 = 10;
	cout << "num1 = " << num1 << endl;
	cout << "num2 = " << num2 << endl;
	cout << "num3 = " << num3 << endl;
	cout << "num4 = " << num4 << endl;

	//2.����sizeof�����������ͻ������ռ�Ŀռ��С
	//sizeof(type or var)����λ���ֽ�
	cout << "short��ռ�Ŀռ䣺" << sizeof(short) << endl;
	cout << "int��ռ�Ŀռ䣺" << sizeof(int) << endl;
	cout << "short num1��ռ�Ŀռ䣺" << sizeof(num1) << endl;

	//3.ʵ��
	//Ĭ������£����һ��С��������ʾ��6λ��Ч����
	float f1 = 3.14f;
	double d1 = 3.14;
	cout << "f1= " << f1 << endl;
	cout << "d1= " << d1 << endl;
	cout << "float��ռ�ռ�" << sizeof(float) << endl;
	cout << "double��ռ�ռ�" << sizeof(double) << endl;
	float f2 = 3e2; //3*10^2
	float f3 = 3e-2;
	cout << "f2= " << f2 << endl;
	cout << "f3= " << f3 << endl;

	//4.�ַ���
	//ֻռһ���ֽڣ��ַ��ͱ��������ǰ��ַ�����ŵ��ڴ��д洢�����ǽ���Ӧ��ASCII������뵽�洢��Ԫ
	char ch1 = 'a';
	cout << "ch1= " << ch1 << endl;
	cout << "char��ռ�ռ� " << sizeof(char) << endl;
	// �ַ��ͱ�����ֵֻ�ܰ���һ���ַ�����ֻ���õ����ţ���ֵ������˫���ţ�"b"
	cout << "ch1='a'��Ӧ��ASCII��ֵ" << (int)ch1 << endl;

	//5.�ַ���
	// Ҫ��˫����
	// a.c���Է���ַ�����
	char ch2[] = "abcdefg";
	// b.c++�����ַ���Ҫ#include<string>,��Щ���ɻ�������Ҫ������ϣ������Ҫд�ϡ�
	//#include <bits/stdc++.h>�޵У�
	string ch3 = "hijklmn";
	cout << "ch2 =" << ch2 << endl;

	//6.bool
	bool flag1 = true;
	bool flag2 = false;
	cout << "flag1 =" << flag1 << endl;// ��ӡ1
	cout << "flag2=" << flag2 << endl;
	cout << "flag2ռ���ڴ�" << sizeof(flag2) << endl; // 1

	//7.��������
	int a = 0;
	cout << "���int����a��ֵ" << endl;
	cin >> a;
	cout << "int����a="<< a << endl;
	return 0;
}