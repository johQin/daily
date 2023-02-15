#include<iostream>
using namespace std;

class Print {
public:
	void operator()(string p) {
		cout << p << endl;
	}
};
int main1() {
	Print p;
	p("hello world");

	Print()("how are you?");

	return 0;
}