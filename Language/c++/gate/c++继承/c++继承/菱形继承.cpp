#include<iostream>
#include<string.h>
using namespace std;
class Animal {
public: 
	//通过static可以解决菱形继承数据多份的问题。
	//static int data;
	int data;
};
// 静态成员使用前必须初始化
//int Animal::data = 100;
class Sheep : public Animal{};
class Tuo : public Animal{};

class SheepTuo:public Sheep,public Tuo{};

int main6() {
	SheepTuo st;
	// 这个好像必须写，暂时不知道原因
	memset(&st, 0, sizeof(SheepTuo));
	
	// 产生了二义性，从羊那继承了一份data，从驼那继承了一份data，所以有两份data
	// 可以通过static和虚继承来解决。
	//cout<<st.data;

	cout << st.Sheep::data<<endl;
	cout << st.Tuo::data << endl;
	return 0;
}