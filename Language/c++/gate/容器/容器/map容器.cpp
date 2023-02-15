#include<map>
#include<string>
#include<iostream>
using namespace std;

class MyGreaterStudent;
class Student {
	friend class MyGreaterStudent;
	friend ostream& operator<<(ostream& out, const Student& stu);
private:
	string name;
	int num;
	float score;
public:
	Student() {};
	Student(string name, int num, float score) {
		this->name = name;
		this->num = num;
		this->score = score;
	}
};
class MyGreaterStudent {
public:
	bool operator()(Student p1, Student p2) const {
		return p1.score > p2.score;
	}
};
ostream& operator<<(ostream& out, const Student& p) {
	out << "num: " << p.num << " name: " << p.name << " score: " << p.score << endl;
	return out;
}

void printMapAll(map<int, Student>& m) {
	map<int, Student>::const_iterator it = m.begin();
	for (; it != m.end(); it++) {
		cout << (*it).first << " " << (*it).second << endl;
	}
}
int main() {
	map<int, Student> m;
	// 方式1
	m.insert(pair<int, Student>(103, Student("john", 103, 80.5f)));
	// 方式2（推荐）
	m.insert(make_pair(104,Student("bob",104,60.5f)));
	// 方式3
	m.insert(map<int, Student>::value_type(105, Student("joe", 105, 70.5f)));
	// 方式4（危险）
	m[106] = Student("lucy", 106, 90.5f);

	printMapAll(m);

	// 当m[107]不存在时，引用m[107]，会在map中新增一个pair<107,>
	//cout << m[107] << endl;
	//printMapAll(m);
	return 0;
}

//