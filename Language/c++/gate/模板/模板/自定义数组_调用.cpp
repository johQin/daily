#include<iostream>
#include<string.h>
#include "自定义可存放任意类型的数组.hpp"
using namespace std;
int main9() {
	MyArray<int> arr;
	arr.pushBack(10);
	arr.pushBack(20);
	arr.pushBack(30);
	arr.pushBack(40);
	arr.pushBack(50);
	arr.toString();
	return 0;
}