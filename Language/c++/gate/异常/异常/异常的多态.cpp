#include<iostream>
using namespace std;
class BaseException {
public:
	virtual void printError() = 0;
};
class NullPointerException :public BaseException {
	void printError() {
		cout << "空指针异常" << endl;
	}
};
class OutOfRangeException :public BaseException {
	void printError() {
		cout << "数组越界异常" << endl;
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