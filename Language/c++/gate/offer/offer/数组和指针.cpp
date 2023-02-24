//#include<vector>
//#include<iostream>
//#include<string>
//using namespace std;
//
//void printVector(vector<int> v1) {
//	// 遍历该容器
//	// 定义一个迭代器iterator,保存起始迭代器
//	vector<int>::iterator it = v1.begin();
//	for (; it != v1.end(); it++) {
//		// it是当前元素的迭代器。可以把it看成是指针，但不要认为it是指针，它底层有很多细节
//		cout << *it << " ";
//	}
//	cout << endl;
//}
//// 1.vector迭代器的使用
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
		cout << "构造函数" << endl;
	}
	~A() {}
	A(const A& other) : str(other.str) {
		cout << "拷贝构造" << endl;
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
		//vec.push_back(i); //调用了10次构造函数和10次拷贝构造函数,
		vec.emplace_back(i); //调用了10次构造函数一次拷贝构造函数都没有调用过
		//vec.emplace_back(A(i));
		//vec.push_back(A(i));
	}
	return 0;
}