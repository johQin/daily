#include<iostream>
#include<string.h>
using namespace std;
class Animal {
public: 
	//ͨ��static���Խ�����μ̳����ݶ�ݵ����⡣
	//static int data;
	int data;
};
// ��̬��Աʹ��ǰ�����ʼ��
//int Animal::data = 100;
class Sheep : public Animal{};
class Tuo : public Animal{};

class SheepTuo:public Sheep,public Tuo{};

int main6() {
	SheepTuo st;
	// ����������д����ʱ��֪��ԭ��
	memset(&st, 0, sizeof(SheepTuo));
	
	// �����˶����ԣ������Ǽ̳���һ��data�������Ǽ̳���һ��data������������data
	// ����ͨ��static����̳��������
	//cout<<st.data;

	cout << st.Sheep::data<<endl;
	cout << st.Tuo::data << endl;
	return 0;
}