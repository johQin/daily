#include<iostream>
using namespace std;


int main3() {
	int a = 10;

	//��const���ε�ָ�������ת��Ϊ ��const
	const int* p1 = &a;
	int* p2 = const_cast<int*>(p1);

	const int& ob=20;
	int& ob1 = const_cast<int&>(ob);


	int b = 20;
	
	//����const���ε�ָ������� ת���� const ��֧�֣�
	int* p3 = &b;
	const int* p4 = const_cast<const int*>(p3);

	int c = 30;
	int& ob2 = c;
	const int& p5 = const_cast<const int&>(ob2);


	return 0;
}