//#include<vector>
//#include<iostream>
//#include<string>
//using namespace std;
//
//void printVector(vector<int> v1) {
//	// ����������
//	// ����һ��������iterator,������ʼ������
//	vector<int>::iterator it = v1.begin();
//	for (; it != v1.end(); it++) {
//		// it�ǵ�ǰԪ�صĵ����������԰�it������ָ�룬����Ҫ��Ϊit��ָ�룬���ײ��кܶ�ϸ��
//		cout << *it << " ";
//	}
//	cout << endl;
//}
//// 1.vector��������ʹ��
//void iteratorUsing() {
//	vector<int> v1;
//	v1.push_back(10);
//	v1.push_back(30);
//	v1.push_back(20);
//	v1.push_back(50);
//	v1.push_back(40);
//	vector<int>::iterator it = v1.begin();
//	printVector(v1);
//	v1.erase(it);
//	printVector(v1);
//	cout << v1.size() << endl;
//	
//}

#include <iostream>
#include <cstring>
#include <vector>
using namespace std;
class A {
public:
	A(int i) {
		str = i;
		cout << "���캯��" << endl;
	}
	~A() {}
	A(const A& other) : str(other.str) {
		cout << "��������" << endl;
	}
	A(A&& other) {
		cout << "in move constructor" << endl;
	}

public:
	int str;
};
int main() {
	vector<A> vec;
	vec.reserve(10);
	for (int i = 0; i < 10; i++) {
		//vec.push_back(i); //������10�ι��캯����10�ο������캯��,
		vec.emplace_back(i); //������10�ι��캯��һ�ο������캯����û�е��ù�
		//vec.emplace_back(A(i));
		//vec.push_back(A(i));
	}
	return 0;
}