#include<iostream>
using namespace std;

class Base {};
class Sub :public Base {};
class Other {};
int main2() {
	//�������ͣ���֧��
	//int num = dynamic_cast<int>(3.14);// error
	//����ת����֧�� ��ȫ
	Base* p1 = dynamic_cast<Base*> (new Sub);
	//����ת������֧�� ������ȫ��
	//Sub* p2 = dynamic_cast<Sub*> (new Base); // error
	//���������ת������֧��
	//Base* p3 = static_cast<Base *> (new Other);// error
	return 0;
}