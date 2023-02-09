#include<iostream>
#include<string.h>
using namespace std;
class Home;
class Mom1 {
public:
    void clean(Home& home);
};
class Home {
    friend class Mom1;
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
void Mom1::clean(Home& home) {
    cout << "Mom need clean bedRoom£º" << home.bedRoom << endl;
}
int main03() {
    Mom1 m;
    Home h("bedRoom01", "livingRoom01");
    m.clean(h);
    return 0;
}