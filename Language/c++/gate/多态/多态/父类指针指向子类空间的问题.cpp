#include<iostream>
using namespace std;
class Animal {
public:
	void speak() {
		cout << "����˵��" << endl;
	}
};
class Dog :public Animal {
public:
	void speak() {
		cout << "����������" << endl;
	}
};
int main1() {
	Animal* b = new Dog;
	b->speak();//����˵�������ǵ������ǵ�������ķ�����������������
	// �����ָ���޷��������෽����λ�á�
	return 0;
}