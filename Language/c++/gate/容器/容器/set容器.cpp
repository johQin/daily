#include<set>
#include<iostream>
using namespace std;

void printSetInt(set<int>& s) {
	set<int>::const_iterator it = s.begin();
	for (; it != s.end(); it++) {
		cout << (*it) << " ";
	}
	cout << endl;
}

// 伪函数的类，MySort()
class MySort {
public:
	//c++ 17 需要在重载这里加const
	bool operator()(int a, int b) const {
		return a > b;
	}
};
void printSetIntSort(set<int,MySort > &s) {
	set<int,MySort>::const_iterator it = s.begin();
	for (; it != s.end(); it++) {
		cout << (*it) << " ";
	}
	cout << endl;
}

class MyGreaterPerson;
class Person {
	friend class MyGreaterPerson;
	friend ostream& operator<<(ostream& out, const Person& stu);
private:
	string name;
	int num;
	float score;
public:
	Person() {};
	Person(string name, int num, float score) {
		this->name = name;
		this->num = num;
		this->score = score;
	}
};
class MyGreaterPerson {
public:
	bool operator()(Person p1, Person p2) const {
		return p1.score > p2.score;
	}
};
ostream& operator<<(ostream& out,const Person& p) {
	out << "num: " << p.num << " name: " << p.name << " score: " << p.score << endl;
	return out;
}

void printSetPerson(set<Person, MyGreaterPerson>& s) {
	set<Person, MyGreaterPerson>::const_iterator it = s.begin();
	for (; it != s.end(); it++) {
		cout << (*it) << " ";
	}
	cout << endl;
}

int main8() {
	set<int> s;
	s.insert(10);
	s.insert(30);
	s.insert(20);
	s.insert(50);
	s.insert(40);
	printSetInt(s);//平衡二叉树，10 20 30 40 50

	//修改排序方式。从大到小，
	// set<int
	set<int, MySort> s1;
	s1.insert(10);
	s1.insert(30);
	s1.insert(20);
	s1.insert(50);
	s1.insert(40);
	printSetIntSort(s1); //50 40 30 20 10

	//set存放自定义数据类型，必须修改排序
	set<Person, MyGreaterPerson> s2;
	s2.insert(Person("john", 101, 80.5f));
	s2.insert(Person("Tom", 102, 70.5f));
	s2.insert(Person("bob", 103, 90.5f));
	s2.insert(Person("joe", 105, 60.5f));
	s2.insert(Person("lucy", 104, 82.5f));
	printSetPerson(s2);
	//num : 103 name : bob score : 90.5
	//num : 104 name : lucy score : 82.5
	//num : 101 name : john score : 80.5
	//num : 102 name : Tom score : 70.5
	//num : 105 name : joe score : 60.5
	return 0;
}