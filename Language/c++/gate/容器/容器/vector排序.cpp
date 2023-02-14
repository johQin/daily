#include<vector>
#include<iostream>
#include<string>
#include<algorithm>
using namespace std;
void printVector03(vector<int>& v) {
	vector<int>::iterator it = v.begin();
	for (; it != v.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}
//使用STL算法对vector容器排序
void vectorIntSort() {
	vector<int> v1;
	v1.push_back(10);
	v1.push_back(30);
	v1.push_back(20);
	v1.push_back(50);
	v1.push_back(40);
	sort(v1.begin(),v1.end());
	printVector03(v1);
}
class Person;
bool comparePerson(Person p1, Person p2);
class Person {
	friend bool comparePerson(Person p1, Person p2);
	friend ostream& operator<<(ostream& out, Person& stu);
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
ostream& operator<<(ostream& out, Person& p) {
	out << "num: " << p.num << " name: " << p.name << " score: " << p.score << endl;
	return out;
}
void printVectorPerson(vector<Person> &v) {
	vector<Person>::iterator it = v.begin();
	for (; it != v.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}
bool comparePerson(Person p1,Person p2) {
	// 从大到小
	return p1.score > p2.score;
}
void vectorPersonSort() {
	vector<Person> v1;
	v1.push_back(Person("john", 101, 80.5f));
	v1.push_back(Person("Tom", 102, 70.5f));
	v1.push_back(Person("bob", 103, 90.5f));
	v1.push_back(Person("joe", 105, 60.5f));
	v1.push_back(Person("lucy", 104, 82.5f));
	sort(v1.begin(), v1.end(), comparePerson);
	printVectorPerson(v1);
}
int main3() {
	//vectorIntSort();
	vectorPersonSort();
	return 0;
}