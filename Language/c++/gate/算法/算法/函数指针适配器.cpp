#include<iostream>
#include<vector>
#include<algorithm>
#include<functional>
using namespace std;

void greaterGate(int val, int gate) {
	cout << "val = " << val << " gate = " << gate << endl;
}
int main5() {
	vector<int> v;
	v.push_back(20);
	v.push_back(10);
	v.push_back(40);
	v.push_back(50);
	v.push_back(30);

	for_each(v.begin(), v.end(), bind2nd(ptr_fun(greaterGate), 40));
	return 0;
}