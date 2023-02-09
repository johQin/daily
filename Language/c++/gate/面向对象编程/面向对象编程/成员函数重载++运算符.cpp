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
	//����++��ob++����Ϊ++����ǰ��ob���ڳ�Ա����������this����һ������ʡ�ԣ��������ǿգ���һ��int��ռλ��
	Student4 operator++(int) {
		Student4 old = *this;
		this->num++;
		this->score++;
		return old;
	}
	// ǰ��++,++ob
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