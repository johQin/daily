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

    // 相互引用
    pb->pa_ = pa;
    pa->pb_ = pb;

    cout << pb.use_count() << endl;//1，weak_ptr没有共享资源，它的构造不会引起指针引用计数的增加。
    cout << pa.use_count() << endl;//2，B里面使用shared_ptr修饰pa_,所以为2
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