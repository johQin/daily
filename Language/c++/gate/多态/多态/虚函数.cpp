#include<iostream>
using namespace std;
class Animal {
public:
	//虚函数
	virtual void speak() {
		cout << "我在说话" << endl;
	}
};
class Dog :public Animal {
public:
	//子类重写父类的虚函数：函数名，返回值类型，参数类型个数顺序，必须完全一致
	void speak() {
		cout << "我在汪汪汪" << endl;
	}
};
class Cat :public Animal {
public:
	//子类重写父类的虚函数：函数名，返回值类型，参数类型个数顺序，必须完全一致
	void speak() {
		cout << "我在喵喵喵" << endl;
	}
};
int main2() {
	Animal* b1 = new Dog;
	b1->speak();//我在汪汪汪

	Animal* b2 = new Cat;
	b2->speak();//我在喵喵喵
	return 0;
}