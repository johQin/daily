#include<iostream>
using namespace std;


class Animal {
public:
	//纯虚函数
	virtual void speak() = 0;
	Animal() {
		cout << "animal 构造" << endl;
	}
	// 虚析构函数，通过父类指针释放子类所有空间。
	// 不用虚析构，那么delete *a的时候，只会释放子类中包含的父类部分空间
	virtual ~Animal() {
		cout << "animal 析构" << endl;
	}
};

class Dog :public Animal {
public:
	void speak() {
		cout << "我在汪汪汪" << endl;
	}
	Dog() {
		cout << "Dog 构造" << endl;
	}
	~Dog() {
		cout << "Dog 析构" << endl;
	}
};

int main5() {
	Animal* a = new Dog;
	a->speak();
	delete a;

	return 0;
}