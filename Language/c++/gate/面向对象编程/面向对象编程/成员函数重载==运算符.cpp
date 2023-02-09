#include<iostream>
using namespace std;
class Student3 {
	friend ostream& operator<<(ostream& out, Student3& stu);
private:
	int num;
	string name;
	float score;
public:
	Student3() {};
	Student3(int num, string name, float score) :num(num), name(name), score(score) {};
	bool operator==(Student3 &stu) {
		if (num == stu.num  && score == stu.score)return true;
		return false;
	}
};


ostream& operator<<(ostream& out, Student3& stu) {
	out << "num: " << stu.num << " name: " << stu.name << " score: " << stu.score << endl;
	return out;
}

int main8() {
	Student3 lucy(100, "lucy", 80.5f);
	Student3 bob(100, "bob", 80.5f);
	cout << lucy << bob << endl;
	
	// ==��ԭ�������
	if (1 == 1) {
		cout << "���"<<endl;
	}
	// ==���غ�Ķ������
	if (lucy == bob) {
		cout << "lucy ���� bob";
	}
	else {
		cout << "lucy������bob";
	}
	return 0;
}