#include<iostream>
using namespace std;
int main() {
	int ret = 0;
	try {
		throw 1;
		//throw 'a';
		//throw 2.14f;
	}
	catch (int e) {
		cout << "int异常值为：" << e << endl;
	}
	// 同类型的异常无法重复捕获
	catch (char e) {
		cout << "char异常值为：" << e << endl;
	}
	catch (...) {
		cout << "其他异常" << endl;
	}
	cout << "------" << endl;
	return 0;
}