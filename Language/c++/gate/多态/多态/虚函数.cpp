#include<iostream>
using namespace std;
class Animal {
public:
	//�麯��
	virtual void speak() {
		cout << "����˵��" << endl;
	}
};
class Dog :public Animal {
public:
	//������д������麯����������������ֵ���ͣ��������͸���˳�򣬱�����ȫһ��
	void speak() {
		cout << "����������" << endl;
	}
};
class Cat :public Animal {
public:
	//������д������麯����������������ֵ���ͣ��������͸���˳�򣬱�����ȫһ��
	void speak() {
		cout << "����������" << endl;
	}
};
int main2() {
	Animal* b1 = new Dog;
	b1->speak();//����������

	Animal* b2 = new Cat;
	b2->speak();//����������
	return 0;
}