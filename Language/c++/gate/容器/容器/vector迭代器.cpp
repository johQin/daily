#include<vector>
#include<iostream>
using namespace std;

void printVector(vector<int>& v);
// 1.vector��������ʹ��
void iteratorUsing() {
	vector<int> v1;
	v1.push_back(10);
	v1.push_back(30);
	v1.push_back(20);
	v1.push_back(50);
	v1.push_back(40);

	printVector(v1);
}

// 2.vector�����µĿռ�
// �����µĿռ��vector����ʼ�������ͻ�仯��
void reCreateSpace() {
	vector<int> v2;
	cout << "������" << v2.capacity() << " ��С��" << v2.size() << endl;// 0 0

	// Ԥ���ռ䣬���¿��Լ����ظ����ٿռ�Ĵ�����
	//v2.reserve(1000);

	vector<int>::iterator it;

	int count = 1;
	it = v2.begin();
	for (int i = 0; i < 1000; i++) {

		v2.push_back(i);

		if (it != v2.begin()) {
			count++;
			cout << "��" << count << "�ο��ٿռ�������" << v2.capacity() << endl;
			it = v2.begin();
		}

	}
}

void printVector(vector<int> & v) {
	// ����������
	// ����һ��������iterator,������ʼ������
	vector<int>::iterator it = v.begin();
	for (; it != v.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}
int main1() {
	//iteratorUsing();
	reCreateSpace();
	return 0;
}