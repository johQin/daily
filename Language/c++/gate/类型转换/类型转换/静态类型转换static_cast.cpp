#include<iostream>
using namespace std;

class Base {};
class Sub:public Base {};
class Other {};
int main1() {
	//�������ͣ�֧��
	int num = static_cast<int>(3.14);
	//����ת����֧�� ��ȫ
	Base * p1 = static_cast<Base *> (new Sub);
	//����ת����֧�� ������ȫ��
	Sub * p2 = static_cast<Sub *> (new Base);
	//���������ת������֧��
	//Base* p3 = static_cast<Base *> (new Other);// error
	return 0;
}