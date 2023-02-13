#include<iostream>
#include "头源分离.h"
// 如果在调用文件中，只include .h文件
// 模板类的运行，需要两次编译，
// 第一次是在预处理阶段，对类模板它本身进行编译
// 第二次是在调用实例化时，这时还要对类模板进行编译，但此时在.h中找不到它的实现，所以无法再次将实际模板类型编译入实现中
// 所以这里就会出现问题，直接报：undefined reference to Data<int,int>::Data()
// 所以这里还需要将.h的实现data.cpp包括进来。
#include "头源分离_类实现.cpp"
using namespace std;
int main7() {
	Data<int, int> ob(10, 20);
	ob.toString();
	return 0;
}