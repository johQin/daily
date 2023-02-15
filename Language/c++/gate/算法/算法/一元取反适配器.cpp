#include<iostream>
#include<vector>
#include<algorithm>
#include<functional>
using namespace std;
int main() {
	vector<int> v;
	v.push_back(10);
	v.push_back(20);
	v.push_back(50);
	v.push_back(30);
	v.push_back(70);
	auto ret=find_if(v.begin(), v.end(), not1(bind2nd(greater<int>(),50)));
	if (ret != v.end()) {
		cout << (*ret) << endl;
	}
	return 0;
}