#include<iostream>
using namespace std;

// �ӿ���
class Animal {
public:
	//���麯��
	virtual void speak() = 0;
};

// ʵ����
class Dog :public Animal {
public:
	void speak() {
		cout << "����������" << endl;
	}
};
class Cat :public Animal {
public:
	void speak() {
		cout << "����������" << endl;
	}
};

// ������
class Speaker {
public:
	//���������Ҫ��������Ϊ�ӿڡ�
	void animalSpeak(Animal* p) {
		p->speak();
	}
};

int main4() {
	Speaker sp;
	sp.animalSpeak(new Dog);
	sp.animalSpeak(new Cat);

	// error��Animal �޷�ʵ����������
	//Animal a;
	return 0;
}