#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

bool greaterThan30(int p){
	return p > 30;
}
class GreaterThan30 {
public:
	bool operator()(int p) {
		return p > 30;
	}
};
int main2() {
	vector<int> v;
	v.push_back(20);
	v.push_back(10);
	v.push_back(40);
	v.push_back(50);
	v.push_back(30);

	vector<int>::iterator ret;
	// 普通函数提供策略，直接函数名
	//ret = find_if(v.begin(), v.end(), greaterThan30);
	// 
	// 仿函数提供策略，需要类名+()
	ret = find_if(v.begin(), v.end(), GreaterThan30());
	
	if (ret != v.end()) {
		cout << "寻找的结果：" << *ret << endl;
	}

	return 0;
}