#include<list>
#include<iostream>
using namespace std;
void printListInt(list<int>& l) {
	list<int>::iterator it = l.begin();
	//list是双向链表，它的迭代器是双向迭代器，不支持+2操作，支持++操作
	for (; it != l.end(); it++) {
		cout << (*it) << " ";
	}
	cout << endl;
}
int main6() {
	list<int> l;
	l.push_back(10);
	l.push_back(30);
	l.push_back(20);
	l.push_back(50);
	l.push_back(40);

	//stl提供的算法 只支持随机访问迭代器，而list是双向迭代器，所以sort不支持
	//sort(l.begin(),l.end());
	l.sort();
	printListInt(l);
	cout << endl;

	
	return 0;
}