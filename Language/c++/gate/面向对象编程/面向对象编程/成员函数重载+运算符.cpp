#include<iostream>
using namespace std;
class Student2 {
	friend ostream& operator<<(ostream& out, Student2& stu);
private:
	int num;
	string name;
	float score;
public:
	Student2() {};
	Student2(int num, string name, float score) :num(num), name(name), score(score) {};
	// 成员函数实现+：lucy+bob ===》lucy.operator+(bob)
	// 所以下面如果定两个形参，将会报错。
	Student2 operator+(Student2& stu2) {
		Student2 stu;
		stu.num = num + stu2.num;
		stu.name = name + stu2.name;
		stu.score = score + stu2.score;
		/*
		stu.num = this->num + stu2.num;
		stu.name = this->name + stu2.name;
		stu.score = this->score + stu2.score;
		*/
		return stu;
	}
};


ostream& operator<<(ostream& out, Student2& stu) {
	out << "num: " << stu.num << " name: " << stu.name << " score: " << stu.score << endl;
	return out;
}

int main7() {
	Student2 lucy(100, "lucy", 80.5f);
	Student2 bob(101, "bob", 90.5f);
	cout << lucy << bob << endl;

	// 成员函数实现+：lucy+bob ===》lucy.operator+(bob)
	//Student2 john = lucy + bob;
	// 二者等价。lucy.operator+ 可以简写为 lucy +
	Student2 john = lucy.operator+(bob);
	cout << john;


	return 0;
}