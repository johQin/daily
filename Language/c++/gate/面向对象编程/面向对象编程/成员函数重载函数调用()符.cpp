#include<iostream>
using namespace std;
class Print {
public:
	//���غ������÷�()
	void operator()(char* str) {
		cout << str << endl;
	}
};
int main10() {
	Print p;
	p("hello world");

	// Print()��������
	Print()("��ã�����");
	return 0;
}