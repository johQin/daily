#include<iostream>
using namespace std;

// �ӿ���
class Animal {
public:
	//�麯��
	virtual void speak() {
		cout << "����˵��" << endl;
	}
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
class Speaker{
public:
	// ��̬�󶨣��ӿ���ָ��
	void animalSpeak(Animal* p) {
		p->speak();
	}
};

int main3() {
	Speaker sp;
	// ��ʲôʵ�����ȥ���͵����Ǹ�ʵ����ķ�����
	sp.animalSpeak(new Dog);
	sp.animalSpeak(new Cat);

	// �Ҳ�֪��������ôȥ��������ռ�
	return 0;
}