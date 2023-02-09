#include<iostream>
using namespace std;
class Student {
	friend ostream& operator<<(ostream& out, Student& stu);
	friend istream& operator>>(istream& in, Student& stu);
private:
	int num;
	string name;
	float score;
public:
	Student() {};
	Student(int num, string name, float score) :num(num), name(name), score(score) {}
};

// 函数的参数为运算符两侧的数据。 
// 由于这个函数需要访问到类的private数据，所以需要用到友元
ostream& operator<<(ostream& out, Student& stu) {
	out << "num: " << stu.num << " name: " << stu.name << " score: " << stu.score << endl;
	//如果需要在其后继续<<endl，那么返回值应是out的引用
	return out;
}
istream& operator>>(istream& in, Student& stu) {
	in >> stu.num >> stu.name  >> stu.score;
	//如果需要在其后继续<<endl，那么返回值应是out的引用
	return in;
}

int main5() {
	Student lucy(100, "lucy", 80.5f);
	Student bob(101, "bob", 90.5f);
	// 如果没有定义重载运算符<<，那么将会报：没有找到接收Student类型的右操作数的运算符（或没有可接受的转换）
	// <<运算符左边不是我们定义的对象，所以采用全局函数来实现运算符的重载。
	//cout << lucy;
	cout << lucy << bob << endl;
	Student john;
	Student joe;
	cin >> john >> joe;
	cout << john << joe;
	return 0;
//num: 100 name : lucy score : 80.5
//num : 101 name : bob score : 90.5
//
//145 john 104.5
//125 joe 100.5
//num : 145 name : john score : 104.5
//num : 125 name : joe score : 100.5
}