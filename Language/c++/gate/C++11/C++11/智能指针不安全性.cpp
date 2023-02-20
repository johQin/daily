#include<iostream>
#include<string>
#include<memory>

using namespace std;
class T1{};
class T2{};
void func(shared_ptr<T1>, shared_ptr<T2>){}
int main2() {
	// ��ʽ1����������ֱ�ӹ�������ָ�루����ȫ��
	shared_ptr<T1> ptr1(new T1());
	shared_ptr<T2> ptr2(new T2());
	func(ptr1, ptr2);

	// main����ִ�в��裺
	//1�������ڴ��T1
	//2�������ڴ��T2
	//3������T1����
	//4������T2����
	//5������T1������ָ�����
	//6������T2������ָ�����
	//7������func

	// ���������ִ�е�3��ʧ�ܣ���ô�ڵ�1,2���������T1��T2���ڴ潫�����й©��
	// ����������ܼ򵥣���Ҫ��shared_ptr���캯����ʹ����������
	// 
	// ��ʽ2������ѡ��make_shared��c++11��/make_unique��c++14������ֱ��ʹ��new������ȫ��
	// ��˵���������ֱ��ʹ��new���ʽ��makeϵ�к����������ŵ㣺�������ظ����롢�Ľ����쳣��ȫ�Ժ����ɵ�Ŀ�����ߴ��С�ٶȸ���

	shared_ptr<T1> ptr3 = make_shared<T1>();
	shared_ptr<T2> ptr4 = make_shared<T2>();

	func(ptr3, ptr4);
	return 0;


}