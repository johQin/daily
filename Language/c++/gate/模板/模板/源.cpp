#include<iostream>
#include<string.h>
using namespace std;

// template ��������������������ֻ�Ե�ǰ������Ч��
template<typename T>
// �����ڵ���ʱ��������������Ƶ���T������
// Ȼ��Ὣ���������е�T�滻��int�����¶Դ����ٱ���һ�Σ�
void swapAll(T& a, T& b) {
	T tmp = a;
	a = b;
	b = tmp;
}
void swapAll(int &a, int &b) {
	int tmp = a;
	
}
int main1() {
	int a = 10, b = 20;
	swapAll(a, b);
	cout << "a = " << a << " b = " << b << endl;
}
