#include<list>
#include<iostream>
using namespace std;
void printListInt(list<int>& l) {
	list<int>::iterator it = l.begin();
	//list��˫���������ĵ�������˫�����������֧��+2������֧��++����
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

	//stl�ṩ���㷨 ֻ֧��������ʵ���������list��˫�������������sort��֧��
	//sort(l.begin(),l.end());
	l.sort();
	printListInt(l);
	cout << endl;

	
	return 0;
}