#include<iostream>
#include<string.h>
using namespace std;
class Home {
    //1.全局函数做友元
    friend void visit01(Home& home);
private:
    string bedRoom;
public:
    string livingRoom;
public:
    Home(string bedRoom, string livingRoom) {
        this->bedRoom = bedRoom;
        this->livingRoom = livingRoom;
    }
};

void visit01(Home& home) {
    cout << "friend visit01 private bedRoom："<<home.bedRoom << endl;
    cout << "friend visit01 public livingRoom：" << home.livingRoom << endl;
}
int main1() {
    Home h("bed01", "living01");
    visit01(h);
    return 0;
}
