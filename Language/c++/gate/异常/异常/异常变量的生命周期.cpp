#include<iostream>
using namespace std;
class MyException {
public:
	MyException() {
		cout << "�쳣�����޲ι���" << endl;
	}
	MyException(const MyException& e) {
		cout << "��������" << endl;
	}
	~MyException() {
		cout << "�쳣��������" << endl;
	}
};
//1.��ͨ���������쳣
void test01(){
	try {
		throw MyException();//����
	}
	catch (MyException e) {// �����������죬Ч���Ե�
		cout << "��ͨ���������쳣" << endl;
	}
}
//2.ָ���������쳣
void test02() {
	try {
		throw new MyException;//����
	}
	catch (MyException *e) {
		cout << "ָ������쳣" << endl;
		delete e;//��������Ҫ��Ϊ�ֶ��ͷš�
	}
}
//3. ���ý����쳣,���������ֶ�
void test03() {
	try {
		throw MyException();//����
	}
	catch (MyException& e) {
		cout << "���ý����쳣" << endl;
	}
}
int main2() {
	test01();
	//�쳣�����޲ι���
	//��������
	//��ͨ���������쳣
	//�쳣��������
	//�쳣��������
	cout<<"-----------" << endl;
	test02();
	//�쳣�����޲ι���
	//ָ������쳣
	//�쳣��������
	cout << "-----------" << endl;
	test03();
	//�쳣�����޲ι���
	//���ý����쳣
	//�쳣��������
	return 0;
}