#include<iostream>
using namespace std;


class Animal {
public:
	//���麯��
	virtual void speak() = 0;
	Animal() {
		cout << "animal ����" << endl;
	}
	// ������������ͨ������ָ���ͷ��������пռ䡣
	// ��������������ôdelete *a��ʱ��ֻ���ͷ������а����ĸ��ಿ�ֿռ�
	virtual ~Animal() {
		cout << "animal ����" << endl;
	}
};

class Dog :public Animal {
public:
	void speak() {
		cout << "����������" << endl;
	}
	Dog() {
		cout << "Dog ����" << endl;
	}
	~Dog() {
		cout << "Dog ����" << endl;
	}
};

int main5() {
	Animal* a = new Dog;
	a->speak();
	delete a;

	return 0;
}