#include<iostream>
#include<string.h>
using namespace std;
class Home;
class Mom {
public:
    void clean(Home& home);
};
class Home {
    friend void Mom::clean(Home& home);
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
void Mom::clean(Home& home) {
    cout << "Mom need clean bedRoom£º" << home.bedRoom << endl;
}
int main2() {
    Mom m;
    Home h("bedRoom01", "livingRoom01");
    m.clean(h);
    return 0;
}