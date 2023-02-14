#include<iostream>
using namespace std;
class BaseException {
public:
	virtual void printError() = 0;
};
class NullPointerException :public BaseException {
	void printError() {
		cout << "��ָ���쳣" << endl;
	}
};
class OutOfRangeException :public BaseException {
	void printError() {
		cout << "����Խ���쳣" << endl;
	}
};
void doWork() {
	throw NullPointerException();
}
int main3() {
	try {
		doWork();
	}
	catch (BaseException& e) {
		e.printError();
	}
	return 0;
}