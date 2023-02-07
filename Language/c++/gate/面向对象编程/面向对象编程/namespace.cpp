#include<iostream>
using namespace std;
namespace A {
	int a = 100;
}
struct Student {
    string name;
    int age;
    void toString() {
        cout << "Name：" << name << endl << "age：" << age << endl;
    }
};
void fun01(void)
{
    cout << "fun01" << endl;
}
void swap(int& x, int& y) {
    int tmp = x;
    x = y;
    y = tmp;
}
void swap01(int* x, int* y) {
    int tmp = *x;
    *x = *y;
    *y = tmp;
}
int main() {
    int a = 10;
    int b = 20;
    cout << "a = " << a << " b = " << b << endl;
    // 传引用
    swap(a, b);
    cout << "a = " << a << " b = " << b << endl;

    int c = 10;
    int d = 20;
    cout << "c = " << c << " d = " << d << endl;
    // 传址
    swap01(&c, &d);
    cout << "c = " << c << " d = " << d << endl;
}