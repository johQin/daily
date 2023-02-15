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
	// bind1st和bind2nd是将二元函数转换为一元函数
	// 比如一个比较大小的函数是二元函数，当在某些情况下我们想要固定第一个参数或者第二个参数时，就成了一元函数

	ret = find_if(v.begin(), v.end(), bind2nd(greater<int>(), 40));
	if (ret != v.end()) {
		cout << "寻找的结果：" << *ret << endl;
	}
	return 0;
}