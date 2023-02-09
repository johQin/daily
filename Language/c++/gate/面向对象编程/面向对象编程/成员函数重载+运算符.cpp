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
	// ��Ա����ʵ��+��lucy+bob ===��lucy.operator+(bob)
	// ������������������βΣ����ᱨ��
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

	// ��Ա����ʵ��+��lucy+bob ===��lucy.operator+(bob)
	//Student2 john = lucy + bob;
	// ���ߵȼۡ�lucy.operator+ ���Լ�дΪ lucy +
	Student2 john = lucy.operator+(bob);
	cout << john;


	return 0;
}