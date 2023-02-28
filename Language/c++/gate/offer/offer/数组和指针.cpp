#include<iostream>
using namespace std;

bool isLittleEndian() {
    unsigned short a = 0x1218;
    cout << &a << endl;
    cout << (char*)&a << endl;
    cout << *(char*)&a << endl;
    if ((*(char*)&a) == 0x18) {
        return true;
    }
    else {
        return false;
    }
}


int main(int argc, char* argv[])
{

    if (isLittleEndian()) {
        cout << "LitteEndian";
    }
    else {
        cout << "BigEndian";
    }

    return 0;
}
