#include<iostream>
using namespace std;


class Animal01 {
public:
	//���麯��
	virtual void speak() = 0;
	//������������������������ʵ��
	virtual ~Animal01() = 0;
};

class Dog :public Animal01 {
public:
	void speak() {
		cout << "����������" << endl;
	}
	~Dog() {
		cout << "Dog ����" << endl;
	}
};

Animal01:: ~Animal01() {
	cout << "Animal01 ����" << endl;
}
int main() {
	Animal01* a = new Dog;
	a->speak();
	delete a;

	return 0;
}