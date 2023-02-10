#include<iostream>
using namespace std;

// 接口类
class Animal {
public:
	//纯虚函数
	virtual void speak() = 0;
};

// 实现类
class Dog :public Animal {
public:
	void speak() {
		cout << "我在汪汪汪" << endl;
	}
};
class Cat :public Animal {
public:
	void speak() {
		cout << "我在喵喵喵" << endl;
	}
};

// 调用类
class Speaker {
public:
	//抽象类的主要作用是作为接口。
	void animalSpeak(Animal* p) {
		p->speak();
	}
};

int main4() {
	Speaker sp;
	sp.animalSpeak(new Dog);
	sp.animalSpeak(new Cat);

	// error：Animal 无法实例化抽象类
	//Animal a;
	return 0;
}