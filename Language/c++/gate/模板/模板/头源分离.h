#pragma once
#include<iostream>
using namespace std;
template<class T1, class T2>
class Data {
private:
	T1 a;
	T2 b;
public:
	Data();
	Data(T1 a, T2 b);
	void toString();
};
