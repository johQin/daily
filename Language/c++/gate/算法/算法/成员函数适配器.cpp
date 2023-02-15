#include<iostream>
#include<vector>
#include<algorithm>
#include<functional>
using namespace std;

class Data {
public:
	int data;
public:
	Data() {};
	Data(int data) {
		this->data = data;
	}
	void printInt(int tmp) {
		cout << "value=" << data + tmp << endl;
	}
};
int main6() {
	vector<Data> v;
	v.push_back(Data(10));
	v.push_back(Data(20));
	v.push_back(Data(50));
	v.push_back(Data(30));
	v.push_back(Data(70));
	//&Data::printInt,成员函数取地址。
	for_each(v.begin(), v.end(), bind2nd(mem_fun_ref(&Data::printInt), 100));


	return 0;
}