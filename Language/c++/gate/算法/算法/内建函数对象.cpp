#include<iostream>
#include<vector>
#include<algorithm>
#include<functional>
using namespace std;
void printAll(vector<int> v1) {
	vector<int>::iterator it = v1.begin();
	for (; it != v1.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}
int main3() {
	vector<int> v;
	v.push_back(20);
	v.push_back(10);
	v.push_back(40);
	v.push_back(50);
	v.push_back(30);

	sort(v.begin(), v.end(), greater<int>());
	printAll(v);//50 40 30 20 10

	vector<int>::iterator ret;
	// bind1st��bind2nd�ǽ���Ԫ����ת��ΪһԪ����
	// ����һ���Ƚϴ�С�ĺ����Ƕ�Ԫ����������ĳЩ�����������Ҫ�̶���һ���������ߵڶ�������ʱ���ͳ���һԪ����

	ret = find_if(v.begin(), v.end(), bind2nd(greater<int>(), 40));
	if (ret != v.end()) {
		cout << "Ѱ�ҵĽ����" << *ret << endl;
	}
	return 0;
}