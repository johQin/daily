#include<iostream>
using namespace std;
class Student1 {
	friend ostream& operator<<(ostream& out, Student1& stu);
	friend Student1 operator+(Student1& stu1, Student1& stu2);
private:
	int num;
	string name;
	float score;
public:
	Student1() {};
	Student1(int num, string name, float score) :num(num), name(name), score(score) {}
};


ostream& operator<<(ostream& out, Student1& stu) {
	out << "num: " << stu.num << " name: " << stu.name << " score: " << stu.score << endl;
	return out;
}
Student1 operator+(Student1& stu1, Student1& stu2) {
	Student1 stu;
	stu.num = stu1.num + stu2.num;
	stu.name = stu1.name + stu2.name;
	stu.score = stu1.score + stu2.score;
	return stu;
}

int main6() {
	Student1 lucy(100, "lucy", 80.5f);
	Student1 bob(101, "bob", 90.5f);
	cout << lucy << bob << endl;

	// 这样写没问题
	Student1 john = lucy + bob;
	cout << john;

	// 可如果这样写,在有的开发环境上会报错，
	// cout << lucy + bob;
	//因为operator+返回的是一个局部匿名对象，而cout的入参是一个对象的引用
	// 局部对象时无法引用的，所以会报错，
	// 所以在重载运算符的时候，像这里的Student1& stu，应该修改为Student1 stu
	
	return 0;
}