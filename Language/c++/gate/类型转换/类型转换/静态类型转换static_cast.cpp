#include<iostream>
using namespace std;

class Base {};
class Sub:public Base {};
class Other {};
int main1() {
	//基本类型：支持
	int num = static_cast<int>(3.14);
	//上行转换：支持 安全
	Base * p1 = static_cast<Base *> (new Sub);
	//下行转换：支持 （不安全）
	Sub * p2 = static_cast<Sub *> (new Base);
	//不相关类型转换：不支持
	//Base* p3 = static_cast<Base *> (new Other);// error
	return 0;
}