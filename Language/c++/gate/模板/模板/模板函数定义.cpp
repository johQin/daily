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
int main2() {
	int a = 10, b = 20;
	swapAll(a, b);
	cout << "a = " << a << " b = " << b << endl;

	string c = "hello", d = "world";
	swapAll(c, d);
	cout << "c = " << c << " d = " << d << endl;
	return 0;
}
