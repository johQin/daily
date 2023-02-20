#include<iostream>
#include<memory>

using namespace std;
int main3() {
    int a = 10;
    shared_ptr<int> ptra = make_shared<int>(a);
    shared_ptr<int> ptr(ptra);     //拷贝构造函数
    cout << ptra.use_count() << endl;   //2

    int b = 20;
    int* pb = &b;

    shared_ptr<int> ptrb = make_shared<int>(b);
    // ptr从ptra转换到ptrb
    ptr = ptrb;
    pb = ptrb.get();    //获取指针

    cout << *pb << endl; // 20, 指向b本身


    cout << ptra.use_count() << endl;   //1，只有ptra引用
    cout << ptrb.use_count() << endl;   //2，ptr和ptrb引用

    return 0;
}