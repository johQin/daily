#include<iostream>
using namespace std;
class Student4 {
	friend ostream& operator<<(ostream& out, Student4& stu);
private:
	int num;
	string name;
	float score;
public:
	Student4() {};
	Student4(int num, string name, float score) :num(num), name(name), score(score) {};
	//后置++，ob++，因为++符号前有ob，在成员函数中属于this（第一个参数省略），后面是空，用一个int来占位。
	Student4 operator++(int) {
		Student4 old = *this;
		this->num++;
		this->score++;
		return old;
	}
	// 前置++,++ob
	Student4 operator++() {
		this->num++;
		this->score++;
		return *this;
	}

};


ostream& operator<<(ostream& out, Student4& stu) {
	out << "num: " << stu.num << " name: " << stu.name << " score: " << stu.score << endl;
	return out;
}

int main9() {
	Student4 lucy(100, "lucy", 80.5f);
	Student4 bob;
	bob = ++lucy;
	cout << bob<<endl;
	cout << lucy << endl;

	return 0;
}