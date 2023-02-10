#include<iostream>
using namespace std;


class Animal01 {
public:
	//纯虚函数
	virtual void speak() = 0;
	//纯虚析构函数，必须在类外实现
	virtual ~Animal01() = 0;
};

class Dog :public Animal01 {
public:
	void speak() {
		cout << "我在汪汪汪" << endl;
	}
	~Dog() {
		cout << "Dog 析构" << endl;
	}
};

Animal01:: ~Animal01() {
	cout << "Animal01 析构" << endl;
}
int main() {
	Animal01* a = new Dog;
	a->speak();
	delete a;

	return 0;
}