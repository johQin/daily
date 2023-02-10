#include<iostream>
using namespace std;

// 接口类
class Animal {
public:
	//虚函数
	virtual void speak() {
		cout << "我在说话" << endl;
	}
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
class Speaker{
public:
	// 动态绑定，接口类指针
	void animalSpeak(Animal* p) {
		p->speak();
	}
};

int main3() {
	Speaker sp;
	// 传什么实现类进去，就调用那个实现类的方法。
	sp.animalSpeak(new Dog);
	sp.animalSpeak(new Cat);

	// 我不知道这里怎么去清理堆区空间
	return 0;
}