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

// �����Ĳ���Ϊ�������������ݡ� 
// �������������Ҫ���ʵ����private���ݣ�������Ҫ�õ���Ԫ
ostream& operator<<(ostream& out, Student& stu) {
	out << "num: " << stu.num << " name: " << stu.name << " score: " << stu.score << endl;
	//�����Ҫ��������<<endl����ô����ֵӦ��out������
	return out;
}
istream& operator>>(istream& in, Student& stu) {
	in >> stu.num >> stu.name  >> stu.score;
	//�����Ҫ��������<<endl����ô����ֵӦ��out������
	return in;
}

int main5() {
	Student lucy(100, "lucy", 80.5f);
	Student bob(101, "bob", 90.5f);
	// ���û�ж������������<<����ô���ᱨ��û���ҵ�����Student���͵��Ҳ����������������û�пɽ��ܵ�ת����
	// <<�������߲������Ƕ���Ķ������Բ���ȫ�ֺ�����ʵ������������ء�
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