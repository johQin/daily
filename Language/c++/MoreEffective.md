# 1 基础议题

## 1. 1 指针与引用

区别：

1. 声明初始化
   - **引用在声明时必须初始化**，不存在未被初始化（指向空值）的引用，在使用前不需要测试它的合法性。
   - 而指针可以不在声明时被初始化，使用前必须检测它的合法性（是否为NULL）
2. 指向变更
   - **引用总是指向在初始化时被指定的对象，以后不能改变。**
   - 指针可以被重新赋值以指向另一个不同的对象

适用场景：

1. 应该使用指针
   - 一是你考虑到存在不指向任何对象的可能（在这种情况下，你能够设置指针为空）
   - 二是你需要能够在不同的时刻指向不同的对象（在这种情况下，你能改变指针的指向）。
2. 使用引用
   - 重载某个操作符时
   - 总是指向一个对象并且一旦指向一个对象后就不会改变指向

```c++
// 引用在一定程度上可以描述为一个“对象常量”
class Widget { ... };
class SpecialWidget: public Widget { ... };
void update(SpecialWidget *psw){
    std::cout<< "123" <<std::endl;
};
SpecialWidget sw;
SpecialWidget& csw = sw;
// 引用取址后就成为了const type * 
// 将一个const type * 转换为type *
update(const_cast<SpecialWidget*>(&csw));
```



## 1.2 尽量使用c++风格的类型转换

```c++
// c的写法
(type) expression
// c++写法
xxx_cast<type>(expression)
    
// 如果你使用的编译器缺乏对新的类型转换方式的支持，你可以用传统的类型转换方法代替 static_cast, const_cast, 以及 reinterpret_cast。也可以用下面的宏替换来模拟新的类型转换语法
#define static_cast(TYPE,EXPR)   ((TYPE)(EXPR))
#define const_cast(TYPE,EXPR)    ((TYPE)(EXPR))

// 这个模拟并不能完全实现 dynamic_cast 的功能，它没有办法知道转换是否失败。
#define dynamic_cast(TYPE,EXPR)  (TYPE)(EXPR)
```

## 1.3 不要对数组使用多态

因为**数组中的每个元素通常都是相同大小的对象**，而派生类对象可能比基类对象更大。这可能导致只能访问到对象的基类部分，而丢失了派生类的特有部分。

```c++
class BST { ... };
class BalancedBST: public BST { ... };

void printBSTArray(ostream& s, const BST array[], int numElements)
{
	for (int i = 0; i < numElements; ++i) {
		s << array[i];
        // array[I]只是一个指针算法的缩写：它所代表的是*(array)。
        // 我们知道 array是一个指向数组起始地址的指针，但是 array 中各元素内存地址与数组的起始地址的间隔究竟有多大呢？它们的间隔是 i*sizeof(一个在数组里的对象)
        // 参数 array 被声明为 BST 类型，所以 array 数组中每一个元素都是 BST 类型，因此每个元素与数组起始地址的间隔是 i*sizeof(BST)
	}
}

BST BSTArray[10];
...
printBSTArray(cout, BSTArray, 10); 			// 运行正常

BalancedBST bBSTArray[10];
...
//你的编译器将会毫无警告地编译这个函数
printBSTArray(cout, bBSTArray, 10);
// 如果你把一个含有 BalancedBST 对象的数组变量传递给 printBSTArray 函数，你的编译器就会犯错误。在这种情况下，编译器原先已经假设数组中元素与 BST 对象的大小一致，但是现在数组中每一个对象大小却与 BalancedBST 一致。派生类的长度通常都比基类要长。我们料想 BalancedBST 对象长度的比 BST 长。如果如此的话，printBSTArray 函数生成的指针算法将是错误的

//

// 一个基类指针来删除一个含有派生类对象的数组，结果将是不确定的
void deleteArray(ostream& logStream, Base array[])
{
	delete [] array;
}
Derived *array = new Derived[50];
deleteArray(cout,array);

```

**数组面对切片分割问题，最好存放对象的指针，而不是对象本身。**



# 2 操作符

## [不同意义的new和delete](https://www.cnblogs.com/area-h-p/p/10345880.html)

1. 如果你想在堆上建立一个对象，应该使用new 操作符，它既分配内存又为对象调用构造函数；
2. 如果你只想分配内存，就调用operator new函数，它不会调用构造函数；
3. 如果你想定制自己的在堆对象被建立时的内存分配过程，你应该写自己的operator new 函数，然后使用new操作符，new操作符会调用你定制的operator new。
4. 如果你想在一块已经获得指针的内存里建立一个对象，应该使用palcement new.



```c++
#include <iostream>


void* operator new(std::size_t size) {
    std::cout << "Custom operator new called\n";
    return std::malloc(size);
}

void operator delete(void* ptr) noexcept {
    std::cout << "Custom operator delete called\n";
    std::free(ptr);
}

class MyClass :public std::string {
public:
    int data;

    MyClass():std::string() {
        std::cout << "Constructor called\n";
    }
    MyClass(std::string a):std::string(a) {
        std::cout << "with params Constructor called\n";
    }

    ~MyClass() {
        std::cout << "Destructor called\n";
    }
    void* operator new(std::size_t size) {
        std::cout << "In Class Custom operator new called\n";
        return std::malloc(size);
    }
    void operator delete(void* ptr) noexcept {
        std::cout << "In Class Custom operator delete called\n";
        std::free(ptr);
    }
    void* operator new(size_t, void *p)//参数size_t没有名字，但是为了防止编译器警告必须加上
    {
        std::cout<< "placement new" <<std::endl;
        return p;
    }
};

int main() {
    //使用类里重载的operator new 为对象MY分配空间
    MyClass * MY = (MyClass*) MyClass::operator new(sizeof(MyClass));
    // 调用placement new函数
    new(MY) MyClass("Hello");
    // 前两行代码等价于在未重载new时 MyClass *MY = new MyClass("Hello");

    // 调用析构函数
    MY->~MyClass();
    //调用operator delete函数
    operator delete (MY);
    // 后两行代码等价于在未重载delete时 delete MY
}

/*
In Class Custom operator new called
placement new
with params Constructor called
Destructor called
Custom operator delete called
*/
```



### new操作（new operator）

```c++
std::string *ps = new std::string("Memory Management");
```

这里的new就是new操作（new operator）。

new 操作符的执行过程：

1. 调用operator new函数分配内存 ；
2. 调用构造函数初始化内存中的对象。

new操作总是会做这两步，不可以任何方式改变。但我们**可以通过operator new来改变步骤1的行为。**

### 操作符new（operator new）

**operator new()完成的操作一般只是分配内存，事实上系统默认的全局::operator new(size_t size)也只是调用malloc分配内存，并且返回一个void*指针。**

```c++
void* operator new(size_t size);		//在做其他形式重载时也要保证第一个参数必须为size_t类型
```

首先，operator new()它是一个函数，并不是运算符。有的地方在运算符重载那里加上了这个，其实它不是运算符。

```c++
#include <iostream>

void* operator new(std::size_t size) {
    std::cout << "Custom operator new called\n";
    return std::malloc(size);
}

void operator delete(void* ptr) noexcept {
    std::cout << "Custom operator delete called\n";
    std::free(ptr);
}

class MyClass {
public:
    int data;

    MyClass() {
        std::cout << "Constructor called\n";
    }

    ~MyClass() {
        std::cout << "Destructor called\n";
    }
    void* operator new(std::size_t size) {
        std::cout << "In Class Custom operator new called\n";
        return std::malloc(size);
    }
    void operator delete(void* ptr) noexcept {
        std::cout << "In Class Custom operator delete called\n";
        std::free(ptr);
    }
};

MyClass * my = (MyClass*) operator new(sizeof(MyClass));			// 该函数调用时与普通函数一样(全局的)
MyClass * inMy = (MyClass*) MyClass::operator new(sizeof(MyClass));	// 类里的
```

对于operator new来说，分为全局重载和类重载，全局重载（类外）是void* operator new(size_t size)，在类中重载形式 void* A::operator new(size_t size)。

operator new就像operator + 一样，是可以重载的。如果类中没有重载operator new，那么调用的就是全局的::operator new来完成堆的分配。同理，operator new[]、operator delete、operator delete[]也是可以重载的。

[**如何限制对象只能建立在堆上或者栈上**](https://blog.csdn.net/wisdomroc/article/details/131455671)

- 只能建立在堆上：设置析构函数为Protected
- 只能建立在栈上：重载new函数设为私有

### 定位new (placement new)

 placement new的作用是在已经被分配但是尚未处理的（raw）内存中构造一个对象。它是一个特殊的operator new

```c++
// 必须写在类里
void* operator new(size_t, void *p)//参数size_t没有名字，但是为了防止编译器警告必须加上
{
      return p;      
}

// 调用placement new函数
new(MY) MyClass("Hello");
// 这初看上去有些陌生，但是它是 new 操作符的一个用法，需要使用一个额外的变量（buffer），当 new 操作符隐含调用 operator new 函数时，把这个变量传递给它。被调用的 operator new函数除了待有强制的参数 size_t 外，还必须接受 void*指针参数，指向构造对象占用的内存空间。这个 operator new 就是 placement new
```

### delete操作与操作符delete

为了避免内存泄漏，每个动态内存分配必须与一个等同相反的 deallocation 对应。函数 operator delete 与 delete 操作符的关系与 operator new 与 new 操作符的关系一样。

```c++
string * ps;
...
delete ps;
```

操作符delete两个过程

1. 调用类的析构函数
2. 调用operator delete释放被对象初始化的内存。

### 数组操作

```c++
string *ps= new string[10];

// 1. 通过调用operator new[]分配数组的内存
// 2. 在数组里的每一个对象的构造函数都必须被调用

// 同样当 delete 操作符用于数组时，它为每个数组元素调用析构函数，然后调用 operator delete 来释放内存。
// 就象你能替换或重载 operator delete 一样，你也替换或重载 operator delete[]。在它们重载的方法上有一些限制。
    
```

# 3 异常

