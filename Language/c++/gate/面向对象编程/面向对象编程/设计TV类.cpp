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
		cout << "电视已开启" << endl;
	}
	if (state == OFF) {
		cout << "电视已关闭" << endl;
	}
	return *this;
}

TV& TV::upChannel()
{
	if (channel >= MAXCHANNEL) {
		cout << "当前已是最大频道："<< MAXCHANNEL << endl;
		return *this;
	}
	channel++;
	cout << "当前频道：" << channel << endl;
	return *this;
}

TV& TV::downChannel()
{
	if (channel <= MINCHANNEL) {
		cout << "当前已是最小频道：" << MINCHANNEL << endl;
		return *this;
	}
	channel--;
	cout << "当前频道：" << channel << endl;
	return *this;
}

TV& TV::upVolumn()
{
	if (volumn >= MAXVOLUMN) {
		cout << "当前已是最大音量：" << MAXVOLUMN << endl;
		return *this;
	}
	volumn++;
	cout << "当前音量：" << volumn << endl;
	return *this;
}

TV& TV::downVolumn()
{
	if (volumn <= MINVOLUMN) {
		cout << "当前已是最小音量：" << MINVOLUMN << endl;
		return *this;
	}
	volumn--;
	cout << "当前音量：" << volumn << endl;
	return *this;
}

TV& TV::showTvInfo()
{
	cout << "频道： " << channel << " 音量： " << volumn << "开启状态： " << (state == ON ? "开启" : "关闭") << endl;
	return *this;
}

int main04() {
	TV tv;
	tv.upChannel().upChannel().showTvInfo();
	return 0;
}