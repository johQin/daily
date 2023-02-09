#include<iostream>
using namespace std;
class TV {
	enum{ ON, OFF };
	enum{ MINCHANNEL = 0, MAXCHANNEL = 10 };
	enum { MINVOLUMN = 0, MAXVOLUMN = 10 };
private:
	int state;
	int channel;
	int volumn;
public:
	TV() {
		state = OFF;
		channel = MINVOLUMN;
		volumn = MINVOLUMN;
	}
	TV& offOrOn();
	TV& upChannel();
	TV& downChannel();
	TV& upVolumn();
	TV& downVolumn();
	TV& showTvInfo();
};

TV& TV::offOrOn()
{
	state = state == OFF ? ON : OFF;
	if (state == ON) {
		cout << "�����ѿ���" << endl;
	}
	if (state == OFF) {
		cout << "�����ѹر�" << endl;
	}
	return *this;
}

TV& TV::upChannel()
{
	if (channel >= MAXCHANNEL) {
		cout << "��ǰ�������Ƶ����"<< MAXCHANNEL << endl;
		return *this;
	}
	channel++;
	cout << "��ǰƵ����" << channel << endl;
	return *this;
}

TV& TV::downChannel()
{
	if (channel <= MINCHANNEL) {
		cout << "��ǰ������СƵ����" << MINCHANNEL << endl;
		return *this;
	}
	channel--;
	cout << "��ǰƵ����" << channel << endl;
	return *this;
}

TV& TV::upVolumn()
{
	if (volumn >= MAXVOLUMN) {
		cout << "��ǰ�������������" << MAXVOLUMN << endl;
		return *this;
	}
	volumn++;
	cout << "��ǰ������" << volumn << endl;
	return *this;
}

TV& TV::downVolumn()
{
	if (volumn <= MINVOLUMN) {
		cout << "��ǰ������С������" << MINVOLUMN << endl;
		return *this;
	}
	volumn--;
	cout << "��ǰ������" << volumn << endl;
	return *this;
}

TV& TV::showTvInfo()
{
	cout << "Ƶ���� " << channel << " ������ " << volumn << "����״̬�� " << (state == ON ? "����" : "�ر�") << endl;
	return *this;
}

int main04() {
	TV tv;
	tv.upChannel().upChannel().showTvInfo();
	return 0;
}