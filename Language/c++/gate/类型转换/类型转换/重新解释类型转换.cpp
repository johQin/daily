#include<iostream>
using namespace std;

class Base {};
class Sub :public Base {};
class Other {};
int main() {
	//�������ͣ���֧��
	//int num = reinterpret_cast<int>(3.14f); //error
	//����ת����֧�� ��ȫ
	Base* p1 = reinterpret_cast<Base*> (new Sub);
	//����ת����֧�� ������ȫ��
	Sub* p2 = reinterpret_cast<Sub*> (new Base);
	//���������ת����֧��
	Base* p3 = reinterpret_cast<Base*> (new Other);
	return 0;
}