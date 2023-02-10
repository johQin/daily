#include<iostream>
using namespace std;
class Animal {
public:
	void speak() {
		cout << "我在说话" << endl;
	}
};
class Dog :public Animal {
public:
	void speak() {
		cout << "我在汪汪汪" << endl;
	}
};
int main1() {
	Animal* b = new Dog;
	b->speak();//我在说话，我们的需求是调用子类的方法，我在汪汪汪。
	// 父类的指针无法到达子类方法的位置。
	return 0;
}