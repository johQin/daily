#include<iostream>
#include<vector>
#include<algorithm>
#include<functional>
using namespace std;
// �ڶ����������̳�binary_function<first_param_type,second_param_type,return_val_type>
class GreaterGate :public binary_function<int, int, void> {
public:
	//��������const���ε�����������غ���
	void operator()(int val, int gate) const{
		cout << "val = " << val << " gate = " << gate << endl;
	}
};
int main4() {
	vector<int> v;
	v.push_back(20);
	v.push_back(10);
	v.push_back(40);
	v.push_back(50);
	v.push_back(30);
	// ��һ��������Ҫ����Ĳ���
	// bind2nd���󶨵�2������Ϊ���������gate��
	for_each(v.begin(), v.end(), bind2nd(GreaterGate(), 40));
	//val = 20 gate = 40
	//val = 10 gate = 40
	//val = 40 gate = 40
	//val = 50 gate = 40
	//val = 30 gate = 40
	// 
	// bind1st���󶨵�1������Ϊ���������gate��
	//for_each(v.begin(), v.end(), bind1st(GreaterGate(), 40));
	
	return 0;
	
}