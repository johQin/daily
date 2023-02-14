#include<deque>
#include<iostream>
using namespace std;
void printDequeInt(deque<int>& d) {
	deque<int>::iterator it = d.begin();
	for (; it != d.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}
int main4() {
	deque<int> d1;
	d1.push_back(1);
	d1.push_back(2);
	d1.push_back(3);
	d1.push_front(4);
	d1.push_front(5);
	d1.push_front(6);
	printDequeInt(d1);
	return 0;
}