#include<vector>
#include<iostream>
using namespace std;

template<typename T>
void printVector(vector<T>& v);
// 1.vector迭代器的使用
void iteratorUsing() {
	vector<int> v1;
	v1.push_back(10);
	v1.push_back(30);
	v1.push_back(20);
	v1.push_back(50);
	v1.push_back(40);
	
	printVector<int>(v1);
}

// 2.vector开辟新的空间
// 开辟新的空间后，vector的起始迭代器就会变化。
void reCreateSpace() {
	vector<int> v2;
	cout << "容量：" << v2.capacity() << " 大小：" << v2.size() << endl;// 0 0
	
	// 预留空间，大致可以减少重复开辟空间的次数。
	//v2.reserve(1000);

	vector<int>::iterator it;

	int count = 1;
	it = v2.begin();
	for (int i = 0; i < 1000; i++) {

		v2.push_back(i);

		if (it != v2.begin()) {
			count++;
			cout << "第" << count << "次开辟空间容量是" << v2.capacity() << endl;
			it = v2.begin();
		}
		
	}
}
template<typename T>
void printVector(vector<T> & v) {
	// 遍历该容器
	// 定义一个迭代器iterator,保存起始迭代器
	vector<T>::iterator it = v.begin();
	for (; it != v.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}
int main() {
	iteratorUsing();
	//reCreateSpace();
	return 0;
}