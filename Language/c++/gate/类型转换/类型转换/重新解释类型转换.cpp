#include<iostream>
using namespace std;

class Base {};
class Sub :public Base {};
class Other {};
int main() {
	//基本类型：不支持
	//int num = reinterpret_cast<int>(3.14f); //error
	//上行转换：支持 安全
	Base* p1 = reinterpret_cast<Base*> (new Sub);
	//下行转换：支持 （不安全）
	Sub* p2 = reinterpret_cast<Sub*> (new Base);
	//不相关类型转换：支持
	Base* p3 = reinterpret_cast<Base*> (new Other);
	return 0;
}