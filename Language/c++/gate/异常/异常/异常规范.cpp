#include<iostream>
using namespace std;
void test() throw() {
	//throw 1;
	throw 'a';
	//throw 2.14f;
	
}
int main1() {
	int ret = 0;
	try {
		test();
	}
	catch (int e) {
		cout << "int�쳣ֵΪ��" << e << endl;
	}
	// ͬ���͵��쳣�޷��ظ�����
	catch (char e) {
		cout << "char�쳣ֵΪ��" << e << endl;
	}
	catch (...) {
		cout << "�����쳣" << endl;
	}
	cout << "------" << endl;
	return 0;
}