#include<iostream>
using namespace std;
class Base2 {
public:
	Base2() {
		cout << "���๹��" << endl;
	}
	~Base2() {
		cout << "��������" << endl;
	}
};
class Member2 {
public:
	Member2() {
		cout << "�����Ա����" << endl;
	}
	~Member2() {
		cout << "�����Ա����" << endl;
	}
};
class Son2 : public Base2 {
public:
	Member2 m;
public:
	Son2() {
		cout << "���๹��" << endl;
	}
	~Son2() {
		cout << "��������" << endl;
	}
};
int main2() {
	Son2 s;

	return 0;
	//���๹��
	//�����Ա����
	//���๹��

	//��������
	//�����Ա����
	//��������
}