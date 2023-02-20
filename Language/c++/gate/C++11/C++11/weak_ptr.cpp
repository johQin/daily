#include<iostream>
#include<memory>
using namespace std;

class B;
class A{
public:
    weak_ptr<B> pb_;
    ~A(){
        cout << "A delete." << endl;
    }
};

class B{
public:
    shared_ptr<A> pa_;
    ~B()
    {
        cout << "B delete." << endl;
    }
};

void fun(){
    shared_ptr<B> pb(new B());
    shared_ptr<A> pa(new A());

    // �໥����
    pb->pa_ = pa;
    pa->pb_ = pb;

    cout << pb.use_count() << endl;//1��weak_ptrû�й�����Դ�����Ĺ��첻������ָ�����ü��������ӡ�
    cout << pa.use_count() << endl;//2��B����ʹ��shared_ptr����pa_,����Ϊ2
}

int main4()
{
    fun();
    return 0;
    //1
    //2
    //B delete.
    //A delete.
}