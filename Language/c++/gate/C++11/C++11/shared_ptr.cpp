#include<iostream>
#include<memory>

using namespace std;
int main3() {
    int a = 10;
    shared_ptr<int> ptra = make_shared<int>(a);
    shared_ptr<int> ptr(ptra);     //�������캯��
    cout << ptra.use_count() << endl;   //2

    int b = 20;
    int* pb = &b;

    shared_ptr<int> ptrb = make_shared<int>(b);
    // ptr��ptraת����ptrb
    ptr = ptrb;
    pb = ptrb.get();    //��ȡָ��

    cout << *pb << endl; // 20, ָ��b����


    cout << ptra.use_count() << endl;   //1��ֻ��ptra����
    cout << ptrb.use_count() << endl;   //2��ptr��ptrb����

    return 0;
}