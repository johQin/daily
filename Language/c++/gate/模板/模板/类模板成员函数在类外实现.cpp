#include<iostream>
using namespace std;

template <class T1, class T2>
class Data {
private:
	T1 a;
	T2 b;
public:
	Data() {}
	Data(T1 a, T2 b);
	void toString();
};

// ����ģ��ֻ�����ڵ�ǰ�࣬���Գ�Ա����������ʵ�֣�������������ģ�塣
// ��ģ��Ĺ�����������ʵ��
template <class T1, class T2>
Data<T1, T2>::Data(T1 a, T2 b)
{
	this->a = a;
	this->b = b;
}

// ��ģ��ĳ�Ա����������ʵ��
template <class T1, class T2>
// ���ܳ�Ա�����Ƿ�Я�Σ�������ʵ�ֶ����븽�����������
// ����ģ�����������дӦ��ΪData<T1,T2>
// ��Ȼ�˴��õ���T1��T2����ô�ͱ��������ǰ�����template
void Data<T1,T2>::toString()
{
	cout << "a = " << a << " b = " << b << endl;
}

int main4() {
	Data<int, int> ob(10, 20);
	ob.toString();
	return 0;
}