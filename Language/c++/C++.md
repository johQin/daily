# C++

C++融合了3种编程模式：面向过程编程（**Process-oriented programming，POP**），面向对象编程（**OOP**），泛型编程。

C++中面向过程的语法兼容大部分C语言，只是在版本不同时，有兼容差异，在一定程度上可以说C++是C语言的超集。

面向对象编程思想：封装，继承，多态

泛型编程思想：模板

# 0 初识

## 0.1 helloworld

使用visual Studio集成开发环境

1. 创建项目

   - <img src="./legend/创建空项目.png" style="zoom: 50%;" />

2. 创建文件

   - ![](./legend/创建helloworld.png)

3. 编写代码

   ```c++
   #include<iostream>
   //包含了目前c++所包含的所有头文件！！！！，有些集成开发环境并不支持，所以也只是一个额外的知识点
   #include<bits/stdc++.h>
   using namespace std;
   
   int main() {
   	//打印字符
   	cout << "Hello world" << endl;
      
       int a = 10;
   	cout << "a=" << a << endl;
   
   	system("pause");
   
   	return 0;
   }
   ```

4. 运行程序

   - ![](./legend/运行helloworld.png)

   

## 0.2 一般知识

1. 关键字

   - 关键字是C++中预先保留的单词（标识符），变量和常量命名不得使用这些关键字

   - |            |              |                  |             |          |
     | ---------- | ------------ | ---------------- | ----------- | -------- |
     | asm        | do           | if               | return      | typedef  |
     | auto       | double       | inline           | short       | typeid   |
     | bool       | dynamic_cast | int              | signed      | typename |
     | break      | else         | long             | sizeof      | union    |
     | case       | enum         | mutable          | static      | unsigned |
     | catch      | explicit     | namespace        | static_cast | using    |
     | char       | export       | new              | struct      | virtual  |
     | class      | extern       | operator         | switch      | void     |
     | const      | false        | private          | template    | volatile |
     | const_cast | float        | protected        | this        | wchar_t  |
     | continue   | for          | public           | throw       | while    |
     | default    | friend       | register         | true        |          |
     | delete     | goto         | reinterpret_cast | try         |          |

5. 标识符命名规则

   - 标识符：只能由字母、数字、下划线组成，并且以字母或下划线为首。
   - 标识符区分大小写

3. 输入输出

   ```c++
   #include<iostream>
   using namespace std;
   #include<bits/stdc++.h>
   int main() {
   
   	//5.字符串
   	// 要用双引号
   	// a.c语言风格字符数组
   	char ch2[] = "abcdefg";
   	// b.c++风格的字符串要#include<string>,有些集成环境不需要，但是希望还是要写上。
   	//#include <bits/stdc++.h>无敌！
   	string ch3 = "hijklmn";
   	cout << "ch2 =" << ch2 << endl;
       cout << "ch3 =" << ch3 << endl;
   
   	//6.bool
   	bool flag1 = true;// 非0的值为真
   	bool flag2 = false;
   	cout << "flag1 =" << flag1 << endl;// 打印1
   	cout << "flag2=" << flag2 << endl;
   	cout << "flag2占用内存" << sizeof(flag2) << endl; // 1
   
   	//7.输入和输出
   	int a = 0;
   	cout << "请给int类型a赋值" << endl;
   	cin >> a;
   	cout << "int类型a="<< a << endl;
   	return 0;
   }
   ```

   

# 1 C++对C的扩展

## 1.1 面向对象编程概述

面向对象编程思想的核心：应对变化，提高复用。

面向对象的三大特点：封装，继承，多态。

在面向对象的语言中：

- 对象 = 算法 + 数据结构
- 程序 = 对象 + 对象 + ...

## 1.2 命名空间namespace

解决命名冲突，用于约束标识符name的作用范围，name可以包括常量，变量，函数，结构体，枚举，类，对象等等。

**作用域运算符：“ :: ”**

```c++
// 1. 定义命名空间
namespace A{
    int a = 100;
}
namespace B{
    int a = 50;
}
void test() {
    cout << "A::a" << A::a << endl; //100
    cout << "B::a" << B::a << endl; //50
}

// 2. 命名空间可嵌套
namespace A {
    int a = 100;
    namespace B{
        int a = 50;
    }
}
void test() {
    cout << "A::a" << A::a << endl; //100
    cout << "A::B::a" << A::B::a << endl; //50
}

// 3. 命名空间是开放的，可以随时添加新成员至已有的命名空间
namespace A{
    int a = 100;
}
namespace A{
    void hello() {
        cout << "hello namespace" << endl;
    }
}
void test() {
    cout << "A::a" << A::a << endl; //100
    A::hello();
}

// 4.声明和实现可分离，可使明明看空间看起来简洁，不臃肿
namespace MySpace{
    void func1();
    void func2(int param);
}
void MySpace::func1(){
    cout << "MySpace::func1" << endl;
}
void MySpace::func2(int param){
    cout << "MySpace::func2" << param << endl;
}

// 5.无名命名空间，相当于给内部成员标识符前面加上static，将成员标识符限制在本文件内
namespace {
    int a = 100;
}

// 6. 命名空间的别名
namespace MyBigMing{
    int a = 100;
}
void home() {
    namespace MySmallMing = MyBigMing;
    cout << "MySmallMing::a" << MySmallMing::a << endl; //100
}
```

## 1.3 using声明

using声明命名空间的成员标识符直接可用

```c++
// 1. using声明成员，直接可用成员名
namespace A {
	int a = 100;
    int b = 50;
}
void test() {
	using A::a;
	cout << "A::a" << a << endl;
    // 没有声明b，b是不能直接使用的
    
    // 相同作用域注意同名冲突
    int a = 50;  // 报错：using 声明导致多次声明“a”	
}

// 2. using声明重载的函数
// 如果命名空间包含相同名字的重载函数，using声明就代表这个重载函数的所有集合
namespace A{
    void func() {};
    void func(int x) {};
    int func(int x, int y){};
}
void test() {
	using A::func;
	func();
    func(1);
    func(1, 2);
}

// 3. using 声明整个命名空间可用
namespace A{
	int param1 = 10;
    float param2 = 1.25f;
    void func1() {};
    int func2(int x) {};
}
void test() {
	using namespace A;
    cout << "A::param1" << param1 << endl;
    cout << "A::param2" << param2 << endl;
    func1();
    func2(1);
}
```

## 1.4 struct类型增强

1. C++中定义结构体变量**不需要**加struct
   - C中，`struct Student s1;`
   - C++中，`Student s1;`
2. C++中成员中可以定义成员函数
   - C中不可以

```c++
struct Student{
    string name;
    int age;
    // 定义成员函数
    void toString() {
        cout << "Name：" << name << endl << "age：" << age << endl;
    }
}
void test() {
    // 定义结构体变量不需要struct
	Student s1;
    s1.name = "qq";
    s1.age = 20;
    s1.toString();
}
```

## 1.5 bool类型

标准c++的bool类型有两种内建的常量true(转换为整数1)和false(转换为整数0)表示状态。大小1byte。

给bool类型赋值时，非0会自动转换为true(1)，0会自动转换为false(0)

```c++
bool b1 = NULL;
bool b2 = '\0';
cout << b1; // 0
cout << b2; // 0

bool flag = true;
cout << flag; // 1
```

## 1.6 引用

在C/C++中指针的作用基本是一样的，但在C++中增加了另外一种给函数传递地址的途径，这就是按引用传递。

C++中新增了 引用的概念，引用可以作为一个已定义变量的别名。

引用的本质就是给变量名取个别名。

定义引用的书写方法步骤：

1. 给谁取别名，就定义谁
2. 然后将定义式从头至尾，将**原名**全部替换为**&别名**

```c++
int a = 10; // 定义式 int a，替换后 int &b
int &b = a; // 初始化
```

>  注意：
>
> 1. 系统不会为引用开辟空间
> 2. 引用和原名代表同一空间的内容
> 3. 操作引用等于操作原名

### 1.6.1 定义引用

```c++
// 1. 普通变量的引用
// 需求：给变量a取个别名叫b
// 定义的时候，&修饰变量为引用，b就是a的别名（引用）
// 系统不会为引用开辟空间，
int a = 10;
int &b = a;// 引用必须初始化

// a和b代表同一空间的内容
cout << "a=" << a << endl; // 10
cout << "b=" << b << endl; // 10
cout << &a << endl; // 0000008B4979F854
cout << &b << endl; // 0000008B4979F854

// 操作b等于操作a
b = 20;
cout << a << endl; // 20
cout << b << endl; // 20


// 2.数组的引用
int arr[5]={10,20,30,40,50};
int n = sizeof(arr)/sizeof(arr[0]);

// 定义数组别名
int (&myArr)[5] = arr;

int i=0;
for(i=0;i<n;i++){
	cout<<myArr[i]<<" ";//10 20 30 40 50
}
cout<<endl;


// 3.指针的引用
int num = 10;
int *p = &num;

int* &myP = p;

cout << "*p = " << *p << endl;//10
cout << "*myP = " << *myP << endl;//10


// 4.函数的引用
void fun01(void)
{
    cout << "fun01" << endl;
}

void(&myFun)(void) = fun01;
myFun();//fun01

// 5.常引用，给常量取别名，给常变量取别名
const int &a = 10;
// int &a = 10; // error
// a = 100; // error
// 不能通过常引用修改内容
// 常引用作为函数的参数，要防止函数内部修改外部的值
```

### 1.6.2 引用作为函数的参数

函数内部可以通过引用操作外部变量

引用作为函数的参数的时

1. 函数调用时传递的实参不必加取址 ”&“ 符
2. 函数定义里面，形参使用时，不必再添加取值“*“符
3. C++主张**引用传递**取代**地址传递**，因为引用语法简单不易出错

```c++
void swap(int &x, int &y) {
    int tmp = x;
    x = y;
    y = tmp;
}
void swap01(int *x, int *y) {
    int tmp = *x;
    *x = *y;
    *y = tmp;
}
int main() {
    int a = 10;
    int b = 20;
    cout << "a = " << a << " b = " << b << endl;
	// 传引用
    swap(a,b);
    cout << "a = " << a << " b = " << b << endl;
    
    int c = 10;
    int d = 20;
    cout << "c = " << c << " d = " << d << endl;
    // 传址
    swap01(&c, &d);
    cout << "c = " << c << " d = " << d << endl;
}
```

### 1.6.3 引用作为函数返回值

不要返回普通局部变量的引用（普通局部变量在使用后，它的内存空间将被释放掉）。

返回值类型为引用，可以完成链式操作

```c++
struct Student{
    Student& printStu(Student &ob, int value){
        cout << value <<;
        return ob;
    }
}
int main() {
    Student obj;
    obj.printStu(obj, 1).printStu(obj, 2).printStu(obj, 3);
}
```

## 1.7 [内联函数](https://blog.csdn.net/nyist_zxp/article/details/119697882)

如果函数是内联的，编译器在编译时，会把内联函数的实现（函数体）替换到每个调用内联函数的地方（函数调用处），避免函数开销。

可以与宏函数作类比，但宏函数不会进行类型检查。

定义时部分inline修饰的函数称为内联函数。

```c++
// 声明的时候不要加inline
int myAdd(int x,int y);
int main() {
    cout << myAdd(100, 200) << endl;
}
inline int myAdd(int x, int y){
    return x + y;
}
```

### 1.7.1 内联函数优缺点

**为什么要使用内联函数：**

引入内联函数主要是解决一些频繁调用的小函数消耗大量空间的问题（减小函数调用栈等函数调用开销）。

通常情况下，在调用函数时，程序会将控制权从调用程序处转移到被调用函数处，在这个过程中，传递参数、寄存器操作、返回值等会消耗额外的时间和内存，如果调用的函数代码量很少，也许转移到调用函数的时间比函数执行的时间更长。而如果使用内联函数，内联函数会在调用处将代码展开，从而节省了调用函数的开销。

**哪些函数不能是内联函数：**

1. 递归调用本身的函数；

2. 包含复杂语句的函数，例如：for、while、switch 等；

3. 函数包含静态变量；

**使用内联函数的缺点：**

1. 如果使用很多内联函数，生成的二进制文件会变大；
2. 编译的时间会增加，因为每次内联函数有修改，就需要重新编译代码。

**内联函数一般要求如下：**

1. 函数简短，通常3-5行
2. 函数内没有复杂的实现，比如：包含while、for 循环，递归等
3. 通常在多处有调用；

**注意事项：**

1. 类中的成员函数默认都是内联函数（不加inline也是内联函数）
2. 有时候，就算加上inline修饰，也不一定是内联函数
3. 有时候，就算没加inline修饰，它也有可能是内联函数
4. 函数是不是内联函数由编译器决定。

### 1.7.2 **宏函数和内联函数的区别**

宏函数和内联函数都会在适当位置进行展开，避免函数调用开销。

参数方面：宏函数的参数没有类型，内联函数的参数有类型，能保证参数的完整性。

处理阶段：宏函数在预处理阶段展开，内联函数在编译阶段展开

作用域：宏函数没有作用域的限制，内联函数有作用域的限制，能作为命名空间，结构体，类的成员。

## 1.8 函数重载

函数重载是C++多态的特性

函数重载：用同一函数名，代表不同的函数功能。

**同一作用域，函数的参数类型不同，个数不同，顺序不同都可以重载。**（返回值类型不同不能作为重载的条件）

```c++
void printFun(int a){
    cout << "int" << endl;
}
void printFun(int a, char b){
    cout << "int char" << endl;
}
void printFun(char b, int a){
    cout << "char int" << endl;
}
void printFun(char a){
    cout << "char" << endl;
}
int main() {
    printFun(10); //int
    printFun(10, 'a'); //int char
    printFun('a', 10); //char int
    printFun('a'); //char
}
```

```c++
// 为什么返回值类型不同不能作为函数重载的条件
// 如果一个函数为 int func(int x)，而另一个函数void fun(int x)，我们直接调用func(10)的时候，就不能确定我们调用的是哪个函数
```

**函数重载底层实现原理**

```c++
void func(){}
void func(int x){}
void func(int x, char y){}
//以上三个函数在linux下编译之后的函数名为：
_zfuncv //v,void
_zfunci //i,int
_zfunciv //i,int,c,char
// 不同的编译器可能产生不同的参数名，这里只做说明
```

## 1.9 函数默认和占位参数

### 1.9.1 默认参数

```c++
void TestFunc01(int a = 10, int b = 20){
cout << "a + b = " << a + b << endl;
}
//注意点:
//1. 形参b设置默认参数值，那么后面位置的形参c也需要设置默认参数
void TestFunc02(int a,int b = 10,int c = 10){}
//2. 如果函数声明和函数定义分开，函数声明设置了默认参数，函数定义不能再设置默认参数
void TestFunc03(int a = 0,int b = 0);
void TestFunc03(int a, int b){}
```

默认参数和函数重载同时出现，一定要注意函数调用的明确性，防止二义性出现

```c++
func(int x){
    cout << "x = " << x << endl;
}
func(int x, int y =20){
    cout << "x = " << x << " y = " << y << endl;
}
int main(){
	// 这里调用明确
    func(100,50);
    // 这里出现二义性，报错
    func(100);
}
```

### 1.9.2 占位参数

c++在声明函数时，可以设置占位参数。**占位参数只有类型名，没有形参名。**

一般情况下，在函数体内无法使用占位参数

占位参数一定要有实参的传递，占位参数也可以拥有默认值，此时可以不传实参

意义：

1. 为以后程序的扩展留下线索
2. 兼容C语言程序中可能出现的不规范写法

```c++
void TestFunc01(int a,int b,int){
	//函数内部无法使用占位参数
	cout << "a + b = " << a + b << endl;
}
//占位参数也可以设置默认值
void TestFunc02(int a, int b, int = 20){
	//函数内部依旧无法使用占位参数
	cout << "a + b = " << a + b << endl;
}
int main(){
	//错误调用，占位参数也是参数，必须传参数
	//TestFunc01(10,20);
    
	TestFunc01(10,20,30);//正确调用
	TestFunc02(10,20);//正确调用
	TestFunc02(10, 20, 30);//正确调用
return 0;
}
```

## 1.10 extern "C"

extern "C"的主要作用就是为了实现C++代码能够调用其他c语言代码。

加上extern "C"后，这部分代码编译器按C语言的方式进行编译和链接，而不是按c++的方式。

```c++
// fun.h
#ifndef MYMODULE_H
#define MYMODULE_H

#include<stdio.h>

#if __cplusplus	// 如果当前是c++工程，下面这段就需要用c语言方式编译
extern "C"{
#endif

    extern void func();
    extern int func2(int a,int b);

#if __cplusplus
}
#endif

#endif
```

```c
// fun.c
#include<stdio.h>
#include "fun.h"
void func(){
    printf("hello c world");
}
int func2(int a,int b){
    return a + b;
}
```

```c++
//main.cpp
#include<iostream>
#inclue "fun.h"
using namespace std;
int main(){
    func1();
    cout << func2(10,20)<<endl;
    return 0;
}
```

## 1.11 new和delete

C++提供了一些关键字，可以按需动态分配内存空间，也可以把不再使用的空间回收再次利用。

静态分配：

1. 在程序编译或运行过程中，按事先规定大小分配内存空间的分配方式，eg：int a[10];
2. 必须事先知道所需空间大小
3. 分配在栈区或全局变量区，一般以数组的形式。

动态分配：

1. 在程序运行过程中，根据需要大小自由分配所需空间
2. 按需分配
3. **分配在堆区，使用一些关键字进行分配**

**new申请堆区空间，delete释放空间。**

**new 和 delete 必须成对存在**

```c++
// 1. 操作基本类型空间

int *p = NULL;
// 从堆区申请int类型大小的空间
p = new int;
*p = 100;
cout << "*p = " << *p << endl;
// 释放空间
delete p;

int *p1 = NULL;
// 申请空间的同时，初始化
p1 = new int(100);
cout << "*p1 = " << *p1 << endl;
delete p1;


// 2. 操作数组空间
int *arr = NULL;
arr = new int[5]{10,20,30,40,50};
for(int i=0;i<5;i++){
    // cout<< *(arr+i) << " ";
    cout << arr[i] << " ";
}
cout<<endl;
delete [] arr;
```



# 2 类和对象

类抽象了事物的属性和行为。

封装性：类将具有共性的数据和方法封装在一起，加以权限区分，用户只能通过公共方法 访问私有数据。

类的权限分为：private（私有）、protected（保护）、public（公有）3种权限。

关于权限的问题，只针对类外。在类的外部，只有public修饰的成员才能被访问，而private、protected都无法被访问。用户在外部可以通过public方法来间接访问private和protected成员数据。

## 2.1 定义类

1. 类中指的是类的定义体里面
2. 类中默认用private修饰成员
3. 类中可以直接使用任意成员变量

```c++
#include <iostream>
using namespace std;
//类Data 是一个类型
class Data{
	//类中 默认为私有
	private:
		int a;//不要给类中成员 初始化
	protected://保护
		int b;
	public://公共
		int c;
		//在类的内部不存在权限之分
		void showData(void)
		{
			cout<<a<<" "<<b<<" "<<c<<endl;
		}
};
int main(){
    //类实例化一个对象
    Data ob;
    //类外不能直接访问 类的私有和保护数据
	//cout<<ob.a <<endl;//err
	//cout<<ob.b <<endl;//err
	cout<<ob.c <<endl;
	//类中的成员函数 需要对象调用
	ob.showData();
}
```

## 2.2 成员函数在类外实现

如果成员函数过多，可以在类外进行实现，这样看起来不臃肿。

实现和声明可以在同文件中，也可以是声明在.h文件中，实现在c或cpp中。

### 同文件中

```c++

class Data{
	private:
		int mA;
	public:
		//类中声明
		void setA(int a);
		int getA(void);
};

void Data::setA(int a){
	mA = a;
}
int Data::getA(){
	return mA;
}
```

### 其他源文件中实现

```c++
// data.h，声明
#ifndef DATA_H
#define DATA_H

class Data{
	private:
		int mA;
	public:
		void setA(int a);
		int getA(void);
};
#endif // DATA_H
```

```c++
//data.cpp，实现
#include "data.h"
void Data::setA(int a){
	mA=a;
}
int Data::getA(){
	return mA;
}
```

```c++
// main.cpp
#include <iostream>
#include "data.h"
using namespace std;
int main(int argc, char *argv[]){
	Data ob;
	ob.setA(100);
	cout<<"mA = "<<ob.getA()<<endl;//mA = 100
	return 0;
}
```

## 2.3 构造与析构函数

创建对象的时候，这个对象与应该一个初始化状态，当对象销毁之前，应该销毁对象自身创建的一些数据。

C++提供了**构造函数和析构函数**，这两个函数会被编译器**自动调用**，完成**对象初始化和对象清理**工作。

即使，我们不提供初始化和清理工作，编译器也会为我们增加默认操作。

### 2.3.1 构造函数

类实例化对象的时候，系统自动调用构造函数，完成对象的初始化工作。

先给对象开辟空间（实例化），然后调用构造函数（初始化）。

如果用户不提供构造函数，编译器会自动添加一个默认的构造函数（空函数）。

构造函数和类名相同，没有返回值类型（连void都不可以有），可以有参数，可以重载，权限必须为public

#### 构造函数的定义

```c++
class Data{
    public:
    	int a;
    public:
    	Data(){
            a = 0;
            cout<<"无参构造器"<<endl;
        }
    	Data(int ma){
            a = ma;
            cout<<"有参构造器 a = "<< ma <<endl;
        }
};
int main() {
    // 隐式调用无参构造
    Data ob1;
    // 显式调用无参构造
    Data ob2 = Data();
    // 隐式调用有参构造
    Data ob3(10);
    // 显式调用有参构造
    Data ob4 = Data(11);
    
	// 调用拷贝构造函数（浅拷贝），后在拷贝构造函数中，会讲到
    Data ob5(ob4)
    
    // 匿名对象，一旦当前语句结束，匿名对象就会被释放
    Data();//调用无参构造
    Data(10);//调用有参构造
    
    // 构造函数的隐式转换，针对类中有单参数构造函数时。
    Data ob6 = 100;// 等价于 Data ob5(100);
}
```

### 2.3.2 析构函数

析构函数名和类名相同，但需要在函数前面加**波浪号~**。

**没有返回值类型，没有函数形参（不能重载）**

当对象的生命周期结束的时候，系统自动调用析构函数。

**先调用析构函数，再释放对象空间。**

析构函数不是必须写的，在没有析构函数的情况下，会调用默认析构函数。

**什么情况下需要写析构函数：**

1. 默认析构函数无法释放我们动态分配的内存（**new出来的**），因此**当存在动态内存分配（以及打开文件）时，要写析构函数释放一下内存。**
2. 类如果存在指针成员，这个类必须写析构函数，释放指针成员所指向的空间
3. 类如果是一个基类，期望用其来派生各种子类。
4. **成员变量有指向堆内存的时候都要写**，默认生成的析构函数只会释放成员变量所占的栈内存，如果存在指针指向一块堆内存是不会被释放的，这就造成了内存泄漏

```c++
class Data{
    public:
    	int a;
    public:
    	Data(){
            a = 0;
            cout<<"无参构造器"<<endl;
        }
    	Data(int ma){
            a = ma;
            cout << "有参构造器 a = " << ma << endl;
        }
    	~Data() {
         	cout << "析构函数 a = " << a << endl;
        }
};

// 查看打印结果
Data ob1(10);
int main() {
    Data ob2(20);
    {
	    Data ob3(30);        
    }
    Data ob4(40);
}
// 构造与释放，在同作用域是符合栈的思想，先进后出，后进先出的思想。
有参构造器 a = 10
    有参构造器 a = 20
    	有参构造器 a = 30
    	析构函数 a = 30
    有参构造器 a = 40
    析构函数 a = 40
    析构函数 a = 20
析构函数 a = 10

```

## 2.4 拷贝构造函数

```c++
// 没有自定义拷贝构造函数时
class Data {
    public:
        int a;
    public:
        Data() {
            cout << "无参构造器" << endl;
        }
        Data(int ma) {
            a = ma;
            cout << "有参构造器：a = " << ma << endl;
        }
        ~Data() {
            cout << "析构函数" << endl;
        }
};
int main() {
    Data ob(10);
    Data ob1 = ob;
}

//命令行打印了，这里可以看到构造函数只调用了一次，而析构函数调用了两次
有参构造器：a = 10
析构函数
析构函数
```



拷贝构造的本质是构造函数。

**拷贝构造函数的调用时机：旧对象 初始化 新对象 才会调用拷贝拷贝构造**

如果用户不提供拷贝构造，编译器会自动提供一个默认的拷贝构造（完成赋值操作——浅拷贝）

一旦定义拷贝构造，系统就会屏蔽系统默认提供的无参构造。

```c++
// 有拷贝构造函数时
class Data {
    public:
        int a;
    public:
        Data() {
            cout << "无参构造器" << endl;
        }
        Data(int ma) {
            a = ma;
            cout << "有参构造器：a = " << ma << endl;
        }
    	// 浅拷贝构造定义形式：ob就是就对象的引用
    	// 如果不写拷贝构造，系统默认也会提供这样的浅拷贝构造
        Data(const Data &ob) {
			// 一旦实现了拷贝构造，必须完成赋值操作，否则a的内存里将出现意想不到的值。并且连浅拷贝的效果都无法实现
            a = ob.a;
            cout << "浅拷贝构造 a = " << a << endl;
        }
        ~Data() {
            cout << "析构函数" << endl;
        }
};
int main() {
    Data ob(10);
    Data ob1 = ob;
}

//命令行打印了，这里可以看到构造函数只调用了一次，而析构函数调用了两次
有参构造器：a = 10
浅拷贝构造 a = 10
析构函数
析构函数
```

### 2.4.1 拷贝构造的调用时机

```c++
// 1.旧对象给新对象初始化时
Data ob1(10);
Data ob2 = ob1;

Data ob3(ob1);

// 2.普通对象作为函数参数，调用函数时，会发生拷贝构造
void func(Data ob){	//Data ob = ob1
    
}
int main() {
    Data ob1(10);
    func(ob1);//此时会调用拷贝构造
}

// 3.函数返回普通对象。有些环境不会发生拷贝构造，而发生了对象接管（整个过程只有一次有参构造和一次析构）
Data func1(){
    Data ob;
    return ob;
}
int main(){
    // 调用拷贝构造，
    func1();// 匿名对象，使用了拷贝构造
    // 和有没有赋值没有关系
    Data ob2 = func1();
}
```

### 2.4.2 深拷贝

默认的拷贝构造都是浅拷贝。

如果类中没有指针成员，则可以不用实现拷贝构造。

如果类中有指针成员，且指向堆区，必须实现析构和深拷贝构造函数。

```C++
class Data1 {
    public:
        char* name;
    public:
  		// 有参构造
        Data1(char* str){
            name = new char[strlen(str) + 1];
            strcpy(name, str);
        }
    	// 拷贝构造
        Data1(const Data1 &ob){
            name = new char[strlen(ob.name) + 1];
            strcpy(name, ob.name);
        }
    	// 析构函数
        ~Data1() {
            if (name != NULL) {
                delete[] name;
                name = NULL;
            }
        }
};
```

## 2.5 [初始化列表](https://blog.csdn.net/gx714433461/article/details/124285721)

类中的成员可以是对象，叫做**对象成员**

一个类在构造对象的时候，会先调用成员对象的构造函数，在调用自身的构造函数。

而一个对象在析构的时候，会先调用自身析构函数，在调用成员对象的析构函数。

构造和析构的调用先后顺序恰好相反。

**类如果想调用对象成员的有参构造，必须使用初始化列表。**

**初始化列表：**

- 以一个冒号开始，接着是一个以逗号分隔的数据成员列表。
- 每个"成员变量"后面跟一个放在括 号中的初始值、参数列表或表达式。

```c++
class Day {
    public:
        int day;
        int hour;
        int minute;
        int second;
    public:
        Day() {
            cout << "day的无参构造" << endl;
        }
        Day(int d,int h, int m, int s) {
            day = d;
            hour = h;
            minute = m;
            second = s;
        }
};
class Date {
    private:
        int _year;
        int _month;
        Day _day;
    public:
    	
	    // 初始化列表
        Date(int year, int month, int day,int h,int m,int s):_month(month),_day(day,h,m,s) {
            _year = year;
        }
    	
    	// 不使用初始化列表
    	//Date(int year, int month, int day, int h, int m, int s) {
        //    // 如果在这里赋值，而不使用初始化列表，那么就会调用对象成员的无参构造函数
        //    _day.day = 12;
        //}
    	
        void printDate() {
            cout << "Date:" << _year << "-" << _month << "-" << _day.day << " " << _day.hour << ":" << _day.minute << ":" << _day.second << endl;
        }
};
int main() {
    Date d(2023, 2, 8, 17, 49, 25);
    d.printDate();
}
```

注意：

  1. 而初始化列表能只能初始化一次。冒号:后面的_month(),或者是_day()只能出现一次，不能多重复。
  2. 编译器允许构造函数赋初值和初始化列表初始化混用，也就是说_month既可以出现在初始化列表中，也可以出现在构造函数中
  3. const成员变量、引用成员变量、没有默认构造函数的自定义类型成员只能在初始化列表初始化。
  4. 成员变量初始化的顺序就是成员变量在类中的声明次序，与初始化列表中的先后次序无关。

## 2.6 [explicit关键字](https://blog.csdn.net/k6604125/article/details/126524992)

C++中的explicit关键字的作用是表明该构造函数是显示的, 而非隐式的。

跟它相对应的另一个关键字是implicit， 意思是隐藏的，类构造函数默认情况下即声明为implicit(隐式)。

 **explicit关键字只对有一个参数的类构造函数有效**，如果类构造函数参数大于或等于两个时, 是不会产生隐式转换的

但是, 也有一个例外，就是当除了第一个参数以外的其他参数都有默认值的时候, explicit关键字依然有效

```c++
class CxString  
{  
    public:  
        char *_pstr;  
        int _size;  
	    // 没有使用explicit关键字的类声明, 即默认为隐式声明  
        CxString(int size)  
        {  
            _size = size;                // string的预设大小  
            _pstr = malloc(size + 1);    // 分配string的内存  
            memset(_pstr, 0, size + 1);  
        }  
        CxString(const char *p)  
        {  
            int size = strlen(p);  
            _pstr = malloc(size + 1);    // 分配string的内存  
            strcpy(_pstr, p);            // 复制字符串  
            _size = strlen(_pstr);  
        }  
        // 析构函数这里不讨论, 省略...  
};  
  
    // 下面是调用:  
  
    CxString string1(24);     // 这样是OK的, 为CxString预分配24字节的大小的内存  
    CxString string2 = 10;    // 这样是OK的, 为CxString预分配10字节的大小的内存  
    CxString string3;         // 这样是不行的, 因为没有默认构造函数, 错误为: “CxString”: 没有合适的默认构造函数可用  
    CxString string4("aaaa"); // 这样是OK的  
    CxString string5 = "bbb"; // 这样也是OK的, 调用的是CxString(const char *p)  
    CxString string6 = 'c';   // 这样也是OK的, 其实调用的是CxString(int size), 且size等于'c'的ascii码  
    string1 = 2;              // 这样也是OK的, 为CxString预分配2字节的大小的内存  
    string2 = 3;              // 这样也是OK的, 为CxString预分配3字节的大小的内存  
    string3 = string1;        // 这样也是OK的, 至少编译是没问题的, 但是如果析构函数里用

```

```c++
class CxString    
{  
public:  
    char *_pstr;  
    int _size;  
    // 使用关键字explicit的类声明, 显示转换
    explicit CxString(int size)  
    {  
        _size = size;  
        // 代码同上, 省略...  
    }  
    CxString(const char *p)  
    {  
        // 代码同上, 省略...  
    }  
}; 

// 下面是调用:  
  
    CxString string1(24);     // 这样是OK的  
    CxString string2 = 10;    // 这样是不行的, 因为explicit关键字取消了隐式转换  
    CxString string3;         // 这样是不行的, 因为没有默认构造函数  
    CxString string4("aaaa"); // 这样是OK的  
    CxString string5 = "bbb"; // 这样也是OK的, 调用的是CxString(const char *p)  
    CxString string6 = 'c';   // 这样是不行的, 其实调用的是CxString(int size), 且size等于'c'的ascii码, 但explicit关键字取消了隐式转换  
    string1 = 2;              // 这样也是不行的, 因为取消了隐式转换  
    string2 = 3;              // 这样也是不行的, 因为取消了隐式转换  
    string3 = string1;        // 这样也是不行的, 因为取消了隐式转换, 除非类实现操作符"="的重载 
```

## 2.7 动态对象的创建



### 2.7.1 c语言方式创建动态对象的问题

c中提供了动态内存的分配，函数malloc，free可以在运行时，从堆中分配存储单元。

然而这些函数在C++中不能很方便的运行，因为它不能帮我们完成对象的初始化工作。

```c++
class Person{
  public:
    int mAge;
    char *pName;
  public:
    Person(){
        mAge=20;
        pName = (char *) malloc(strlen("john") + 1);
        strcpy(pName, "john");
    }
    void Init(){
        mAge=20;
        pName = (char *) malloc(strlen("john") + 1);
        strcpy(pName, "john");
    }
    void Clean(){
        if(pName != NULL){
            free(pName);
        }
    }
};
int main(){
    //分配内存
    Person *person = (Person *) malloc(sizeof(Person));
    if(person == NULL){
        return 0;
    }
    // 调用初始化函数，需要手动去初始化
    person->Init();
    // 清理对象
    person->Clean();
    // 释放person空间
    free(person);
    return 0;
}
```

问题：

1. 必须知道对象的长度（sizeof）
2. malloc后，必须强转为指针
3. malloc后，可能内存申请失败
4. 在使用对象之前，必须记住为它初始化（Init)

### 2.7.2 new创建和delete释放动态对象

当用new创建一个对象时，它就在堆区里为对象**分配内存并调用构造函数完成初始化**。

而delete表达式先调用析构函数，然后释放内存。

只需要一个简单的new表达式，它带有内置的长度计算，类型转换和安全检查。

这样在堆里创建对象可以想在栈里创建对象一样简单。

```c++
class Person {
    public:
        char* pName;
        int mAge;
    public:
        Person() {
            cout << "无参构造" << endl;
            pName = new char[strlen("undefined") + 1];
            strcpy(pName, "undefined");
            mAge = 0;
        }
        Person(char *name, int age) {
            cout << "有参构造" << endl;
            pName = new char[strlen(name) + 1];
            strcpy(pName, name);
            mAge = age;
        }
        ~Person() {
            cout << "析构" << endl;
            if (pName != NULL) {
                delete[] pName;
                pName = NULL;
            }
        }
        void showPersonInfo() {
            cout << "Name: " << pName << " age: " << mAge << endl;
        }
};
int main() {
    Person* person1 = new Person;
    Person* person2 = new Person("john", 25);
    person1->showPersonInfo();
    person2->showPersonInfo();
    
}
```

### 2.7.3 动态对象数组

```c++
Person* persons = new Person[20];
delete [] persons;
```

## 2.8 静态成员

在类定义中，它的成员（包括成员变量和成员函数），这些成员可以用**关键字static声明为静态的，称为 静态成员**。 

不管这个类创建了多少个对象，**静态成员只有一个拷贝**，

静态成员在类的所有对象中是共享的。

static修饰的成员属于类，而非对象，先于对象存在。

static修饰的成员，在定义类的时候，必须分配空间。

静态成员：在成员变量和成员函数前加上关键字static。

静态成员分为：静态成员变量和静态成员函数

静态成员变量：

1. 在编译阶段分配内存。
2. 类内声明，类外初始化（C++11支持类中初始化)。
3. 所有对象共享同一份数据。

静态成员函数：

1. 所有对象共享同一个函数。
2. 静态成员函数没有 this 指针，只能访问静态成员变量。普通成员函数有 this 指针，可以访问类中的任意成员

静态成员的两种访问方式：

- 通过对象（A a; a.b;a.func）。
- 通过类名（A::b;A::func）

**静态成员变量使用前必须先初始化(如int MyClass::m_nNumber = 0;)，否则会在linker时出错。**

**非静态可以访问静态，静态无法访问非静态。**

```c++
class Data{
  public:
    int a;
    // 类中定义
    static int b;//静态成员变量
  public:
    static int addB(){
        //只能访问静态成员
        return b+=10;
    }
};
// 类外初始化
int Data::b = 10; // 前面需要加类型符int，否则会报：缺少类型说明符
int main(){
    Data ob1;
    //对象访问静态成员变量
    cout<<ob1.b<<endl;// 10
    //类名访问静态成员
    cout<<Data::b<<endl;//10
    Data::addB();
    cout<<Data::b<<endl;//20
    
    ob1.b = 30;
    cout<<Data::b<<endl;//30
    Data ob2;
    ob2.b = 40;
    cout<<Data::b<<endl;//40
}
```

## 2.9 this指针

成员变量和成员函数时分开存储的。

C++中，非静态成员变量是直接内含在类对象中，**静态成员和函数都不占对象空间。**

成员函数是独立存储的，成员函数只有一份，所有对象共享。

sizeof(ClassName)——计算的是类对象所占的空间大小。

由于成员函数只有一份，函数需要知道是哪一个对象在调用自己。

**this指针指向当前对象（正在调用成员函数的对象）**

成员函数通过this指针即可知道当前正在操作哪个对象的数据。**this指针是一种隐含指针，它隐含于每个类的非静态成员函数中**，this指针无需定义，直接使用

```c++
class Data {
public:
    int ma;
    int b;
public:
    // 隐含this
    void setMa(int a) {
        ma = a; // 这里隐含了this指针，实际为this->ma = a;
    }
    int getMa() {
        cout << "ma = " << ma << endl; // cout << this->ma <<endl
        return ma;
    }
    // this指针应用1：当形参和成员变量名相同时，做区分
    void setB(int b) {
        this->b = b;// 显式this
    }
    int getB() {
        cout << "b = " << this->b << endl;
        return this->b;
    }
    // this指针应用2：做链式操作
    Data& printField() {
        cout << "ma = " << ma << " b = " << b << endl;
        return *this;
    }
};
int main() {
    Data ob;
    
    // 隐式this操作
    ob.setMa(10);
    ob.getMa();
    
    // 显式this操作
    ob.setB(20);
    ob.getB();
    
	// 链式操作
    ob.printField().printField().printField();
}
```

### const修饰成员函数

const修饰成员函数时，const会实际应用到this指针指向的内存区域，

也就是说**成员函数体内不可修改对象中的任何非静态成员变量，但是可以修改静态成员变量和mutable修饰的成员变量。**

```c++
class Data{
  public:
    int ma;
    mutable int b;
    static int c;
  public:
    void setMa(int a) const {
        ma = a; // 出错，表达式必须是可修改的左值
        c = 100;// ok，但在使用c前，必须初始化，否则会报错
        b=50;// ok
    }
};
int Data::c = 10;
int main() {
    Data ob;
    ob.setMa();
    return 0;
}
```

## 2.10 友元

类的私有成员无法在类的外部访问，但是有时候，需要在类的外部访问私有成员，友元就是用来解决这个问题。友元破坏了封装性。

程序员可以把一个全局函数，某个类中的成员函数、甚至整个类声明为友元。

### 2.10.1友元语法

使用friend关键字声明友元，friend关键字只出现在声明处。

一个函数或者类，作为了另一个类的友元，那么这个函数或者类就可以直接访问另个类的所有数据（包括私有）。

类似于类里声明了一个白名单，告诉编译器哪些函数，哪些类可以访问我的私有数据。

友元的重要应用在运算符重载上。

#### 全局函数做类的友元

```c++
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
}

```

#### [类的public函数做友元](https://blog.csdn.net/qq_43259304/article/details/89605118?spm=1001.2014.3001.5502)

在同一个cpp里面写两个类，并且存在友元关系的时候，**尤为注意书写顺序。**

B::bfunc做A的友元函数

1. A需要在B的上方，前向声明类名。因为B中会用到类名A
2. B::bfunc需要在A前面做声明。否则friend那里无法识别B::func
3. B::bfunc的实现需要放在A定义的后面，因为B::bfunc需要使用到A及其成员。

```c++
// 这里要尤为注意，Home和Mom两个类的定义位置，

#include<iostream>
#include<string.h>
using namespace std;

//前向声明，让Mom的clean声明知道有Home这个类
class Home;


class Mom {
public:
    void clean(Home& home);
    // 这里不能对clean进行实现，因为clean里面用到了Home的bedRoom字段，
    // 前向声明类名是无法让编译器知道有bedRoom这个字段的
    // 所以这个函数的实现只能放到Home声明的后面
    
};

// Mom这个类声明了，下面的Mom::clean才会识别到
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

// 因为这个实现里面使用到了Home的bedRoom，所以实现需要放在Home类的后面
void Mom::clean(Home& home) {
    cout << "Mom need clean bedRoom：" << home.bedRoom << endl;
}
int main() {
    Mom m;
    Home h("bedRoom01", "livingRoom01");
    m.clean(h);
    return 0;
}
```

#### 类做友元

```c++
#include<iostream>
#include<string.h>
using namespace std;

class Home;

class Mom1 {
public:
    void clean(Home& home);
};

class Home {

    
    // 整个类做友元
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
    cout << "Mom need clean bedRoom：" << home.bedRoom << endl;
}


int main() {
    Mom1 m;
    Home h("bedRoom01", "livingRoom01");
    m.clean(h);
    return 0;
}
```

### 2.10.2 友元注意事项

1. **友元关系不能被继承。**
2. **友元关系是单向的。**类A是类B的朋友，但类B不一定是类A的朋友。
3. **友元关系不具有传递性。**类B是类A的朋友，类C是类B的朋友，但类C不一定是类A的朋友

## 2.11 运算符重载

运算符重载，就是对已有的运算符重新进行定义，赋予其另一种功能，以适应不同的数据类型。

运算符的操作数类型不同，会使用不同的定义，所以这里叫做**运算符的重载**，和函数重载差不多是一个意思。

语法： 定义重载的运算符就像定义函数，只是该函数的名字是operator@，这里的@代表了被重载的运算符。

思路：

1. 弄懂运算符的运算对象的个数。（个数决定了 重载函数的参数个数）
2. 识别运算符左边的运算对象 是类的对象 还是其他. 
   - 类的对象：全局函数实现（不推荐） 成员函数实现（推荐，少一个参数）
   -  其他：只能是全局函数实现

如果使用全局函数 重载运算符 必须将全局函数设置成友元。

如果减号由成员函数实现重载，**`ob1.operator-(ob2)`等价于 `ob1 - ob2`，**

可以重载的运算符，其中不要重载&&和||，因为我们无法实现它的短路特性。

![](./legend/可以重载的运算符.jpeg)

### 2.11.1 [重载输出输入运算符](https://blog.csdn.net/lu_embedded/article/details/121599696)

全局函数重载输出运算符

```c++

#include<iostream>
using namespace std;
class Student {
    //添加友元函数
	friend ostream& operator<<(ostream& out, Student& stu);
	friend istream& operator>>(istream& in, Student& stu);
private:
	int num;
	string name;
	float score;
public:
	Student() {};
	Student(int num, string name, float score) :num(num), name(name), score(score) {}
};

// 函数的参数为运算符两侧的数据。 
// 由于这个函数需要访问到类的private数据，所以需要用到友元
ostream& operator<<(ostream& out, Student& stu) {
	out << "num: " << stu.num << " name: " << stu.name << " score: " << stu.score << endl;
	//如果需要在其后继续<<endl，那么返回值应是out的引用
	return out;
}
istream& operator>>(istream& in, Student& stu) {
	in >> stu.num >> stu.name  >> stu.score;
	//如果需要在其后继续<<endl，那么返回值应是out的引用
	return in;
}

int main() {
	Student lucy(100, "lucy", 80.5f);
	Student bob(101, "bob", 90.5f);
	// 如果没有定义重载运算符<<，那么将会报：没有找到接收Student类型的右操作数的运算符（或没有可接受的转换）
	// <<运算符左边不是我们定义的对象，所以采用全局函数来实现运算符的重载。
	//cout << lucy;
	cout << lucy << bob << endl;
	Student john;
	Student joe;
	cin >> john >> joe;
	cout << john << joe;

//num: 100 name : lucy score : 80.5
//num : 101 name : bob score : 90.5
//
//145 john 104.5
//125 joe 100.5
//num : 145 name : john score : 104.5
//num : 125 name : joe score : 100.5
}
```

### 2.11.2 重载"+"运算符

#### 全局函数实现重载+

```c++

#include<iostream>
using namespace std;
class Student1 {
	friend ostream& operator<<(ostream& out, Student1& stu);
	friend Student1 operator+(Student1& stu1, Student1& stu2);
private:
	int num;
	string name;
	float score;
public:
	Student1() {};
	Student1(int num, string name, float score) :num(num), name(name), score(score) {}
};


ostream& operator<<(ostream& out, Student1& stu) {
	out << "num: " << stu.num << " name: " << stu.name << " score: " << stu.score << endl;
	return out;
}
Student1 operator+(Student1& stu1, Student1& stu2) {
	Student1 stu;
	stu.num = stu1.num + stu2.num;
	stu.name = stu1.name + stu2.name;
	stu.score = stu1.score + stu2.score;
	return stu;
}

int main() {
	Student1 lucy(100, "lucy", 80.5f);
	Student1 bob(101, "bob", 90.5f);
	cout << lucy << bob << endl;

	// 这样写没问题
	Student1 john = lucy + bob;
	cout << john;

	// 可如果这样写,在有的开发环境上会报错，
	// cout << lucy + bob;
	//因为operator+返回的是一个局部匿名对象，而cout的入参是一个对象的引用
	// 局部对象时无法引用的，所以会报错，
	// 所以在重载运算符的时候，像这里的Student1& stu，应该修改为Student1 stu
	
	return 0;
}
```

#### 成员函数实现重载+

```c++
#include<iostream>
using namespace std;
class Student2 {
	friend ostream& operator<<(ostream& out, Student2& stu);
private:
	int num;
	string name;
	float score;
public:
	Student2() {};
	Student2(int num, string name, float score) :num(num), name(name), score(score) {};
	// 成员函数实现+：lucy+bob ===》lucy.operator+(bob)
	// 所以下面如果定两个形参，将会报错。
	Student2 operator+(Student2& stu2) {
		Student2 stu;
		stu.num = num + stu2.num;
		stu.name = name + stu2.name;
		stu.score = score + stu2.score;
		/*
		stu.num = this->num + stu2.num;
		stu.name = this->name + stu2.name;
		stu.score = this->score + stu2.score;
		*/
		return stu;
	}
};


ostream& operator<<(ostream& out, Student2& stu) {
	out << "num: " << stu.num << " name: " << stu.name << " score: " << stu.score << endl;
	return out;
}

int main() {
	Student2 lucy(100, "lucy", 80.5f);
	Student2 bob(101, "bob", 90.5f);
	cout << lucy << bob << endl;

	// 成员函数实现+：lucy+bob ===》lucy.operator+(bob)
	//Student2 john = lucy + bob;
	// 二者等价。lucy.operator+(bob) 可以简写为 lucy + bob
	Student2 john = lucy.operator+(bob);
	cout << john;


	return 0;
}
```

### 2.11.3 重载逻辑运算符==

```c++
#include<iostream>
using namespace std;
class Student3 {
	friend ostream& operator<<(ostream& out, Student3& stu);
private:
	int num;
	string name;
	float score;
public:
	Student3() {};
	Student3(int num, string name, float score) :num(num), name(name), score(score) {};
    
    
	bool operator==(Student3 &stu) {
		if (num == stu.num  && score == stu.score)return true;
		return false;
	}
    
    
};


ostream& operator<<(ostream& out, Student3& stu) {
	out << "num: " << stu.num << " name: " << stu.name << " score: " << stu.score << endl;
	return out;
}

int main() {
	Student3 lucy(100, "lucy", 80.5f);
	Student3 bob(100, "bob", 80.5f);
	cout << lucy << bob << endl;
	
	// ==的原定义操作
	if (1 == 1) {
		cout << "你好"<<endl;
	}
	// ==重载后的定义操作
	if (lucy == bob) {
		cout << "lucy 等于 bob";
	}
	else {
		cout << "lucy不等于bob";
	}
	return 0;
}
```

### 2.11.4 重载++和--

由于该操作符有两种运算，分别为前置++和后置++，**通过占位参数的方法去区分这种操作。**

- ++a(前置++)，它就调用operator++(a)；
- a++（后置++），它就会去调用operator++(a,int)；

```c++
#include<iostream>
using namespace std;
class Student4 {
	friend ostream& operator<<(ostream& out, Student4& stu);
private:
	int num;
	string name;
	float score;
public:
	Student4() {};
	Student4(int num, string name, float score) :num(num), name(name), score(score) {};
	//后置++，ob++，因为++符号前有ob，在成员函数中属于this（第一个参数省略），后面是空，用一个int来占位。
	Student4 operator++(int) {
		Student4 old = *this;
		this->num++;
		this->score++;
		return old;
	}
	// 前置++,++ob
	Student4 operator++() {
		this->num++;
		this->score++;
		return *this;
	}

};


ostream& operator<<(ostream& out, Student4& stu) {
	out << "num: " << stu.num << " name: " << stu.name << " score: " << stu.score << endl;
	return out;
}

int main() {
	Student4 lucy(100, "lucy", 80.5f);
	Student4 bob;
	bob = ++lucy;
	cout << bob<<endl;
	cout << lucy << endl;

	return 0;
}
```

### 2.11.5 重载函数调用符"()"

重载"()"运算符一般用于为算法提供策略

当对象和()结合会触发重载函数运算调用运算符。

```c++
#include<iostream>
using namespace std;
class Print {
public:
	//重载函数调用符()
	void operator()(char* str) {
		cout << str << endl;
	}
};
int main() {
	Print p;
    // 伪函数
	p("hello world");

	// Print()匿名对象，
	Print()("你好，世界");
	return 0;
}
```

### 2.11.6 重载”->和*“——智能指针实现堆区空间自动释放

智能指针实现堆区空间自动释放。

```c++
#include<iostream>
using namespace std;
class Data{
public:
	Data() {
		cout << "无参构造" << endl;
	}
	~Data() {
		cout << "析构" << endl;
	}
	void func() {
		cout << "func" << endl;
	}
};
class SmartPointer {
public:
	Data* p;
public:
	SmartPointer(Data* p) {
		this->p = p;
	}
	~SmartPointer() {
		delete p;
	}
	Data* operator->() {
		return p;
	}
	Data& operator*() {
		return *p;
	}
};
int main() {
	/*
	Data* p = new Data;
	p->func();
	delete p;
	*/

	SmartPointer sp(new Data);
	// 在->没重载之前，是这样调用func的。
	//sp.p->func();
	// ->重载之后
	sp->func();

	// *重载后
	(*sp).func();
	return 0;
}
```

# 3 继承

c++最重要的特征是代码重用。

通过继承机制可以利用已有的数据类型来定义新的数据类型，新的类不仅拥有旧类的成员，还拥有新定义的成员。

## 3.1 继承与派生

```c++
class A{};
class B : extend_type A{};
```

继承方式`extend_type`有三种：

1. 公共继承public，父类成员的修饰方式保持不变
2. 保护继承protected，父类成员的修饰方式变保护
3. 私有继承private，父类成员的修饰方式变私有

**所有父类的私有成员在子类中不可访问，私有依旧私有。**

<img src="./legend/继承方式.png" style="zoom:67%;" />

```c++
#include<iostream>
using namespace std;
class Base {
public:
	int a;
protected:
	int b;
private:
	int c;
};
class Son : public Base {
public:
	void func() {
		cout << "a = " << a << " b = " << b << endl;
		//cout << "c = " << c << endl;
	}
};
int main() {
	Son s;
	s.func();
	//类外无法访问
	//cout << s.b << endl;
	//cout << s.c << endl;
}
```

## 3.2 继承中的构造与析构

### 3.2.1 子类构造析构顺序

<img src="./legend/继承中的构造与析构顺序.png" style="zoom:67%;" />

```c++
#include<iostream>
using namespace std;
class Base2 {
public:
	Base2() {
		cout << "父类构造" << endl;
	}
	~Base2() {
		cout << "父类析构" << endl;
	}
};
class Member {
public:
	Member() {
		cout << "子类成员构造" << endl;
	}
	~Member() {
		cout << "子类成员析构" << endl;
	}
};
class Son2 : public Base2 {
public:
	Member m;
public:
	Son2() {
		cout << "子类构造" << endl;
	}
	~Son2() {
		cout << "子类析构" << endl;
	}
};
int main() {
	Son2 s;

	return 0;
	//父类构造
	//子类成员构造
	//子类构造

	//子类析构
	//子类成员析构
	//父类析构
}
```

### 3.2.2 子类调用成员对象、父类的有参构造

子类实例化对象时，会自动调用成员对象和父类的默认构造。

子类实例化对象是，如果想要调用成员对象和父类的有参构造，则必须使用初始化列表。

初始化列表中：父类写类名称，成员对象用对象名。

```c++
#include<iostream>
using namespace std;
class Base3 {
public:
	int a;
public:
	Base3() {
		cout << "父类构造" << endl;
	}
	Base3(int a) {
		this->a = a;
		cout << "父类有参构造" << endl;
	}

};
class Member3 {
public:
	int b;
public:
	Member3() {
		cout << "子类成员构造" << endl;
	}
	Member3(int b) {
		this->b = b;
		cout << "子类成员有参构造" << endl;
	}
};
class Son3 : public Base3 {

public:
	int c;
	Member3 m;
public:
	Son3() {
		cout << "子类构造" << endl;
	}
	Son3(int a, int b, int c) :Base3(a),m(b),c(c){
		cout << "子类有参构造" << endl;
	}
};
int main() {
	Son3 s(1,2,3);
	return 0;

}
```

## 3.3 子类成员重名

同名成员最安全最有效的处理方式：加作用域**::**

```c++
#include<iostream>
#include<string.h>
using namespace std;
class Base4 {
public:
	int a;
	string b;
	Base4(int a) {
		this->a = a;
		b = "父类b";
	}
	void func() {
		cout << "父类func：a = " << a << endl;
	}
	void func01() {
		cout << "父类func01：b = " << b << endl;
	}
};
class Son4 : public Base4 {
public:
	int a;
	Son4(int x, int y) :Base4(x) {
		a = y;
	}
	void func() {
		cout << "子类func：a = " << a << endl;
	}
};
int main() {
	Son4 s(10,20);

	// 子类默认优先查找自身类有没有对应的成员
	// 如果有则直接返回
	// 如果没有则继续向上在父类中查找
	cout<<s.a<<endl; // 20
	cout << s.b << endl; //父类b
	s.func();// 子类func
	s.func01();// 父类func01

	// 重名时，加作用域区分，即可正确使用父类对应的成员
	cout<<s.Base4::a<<endl;
	s.Base4::func();
	return 0;
}
```

### 3.3.1 子类定义父类中同名有重载的函数

重载：是没有继承的。

重定义：是有继承的。

子类一旦定义了父类同名并且父类中有重载的函数，子类中都将屏蔽父类的所有同名函数。

子类可以通过作用域来访问父类中的同名重载函数。

```c++
#include<iostream>
#include<string.h>
using namespace std;
class Base5 {
public:
	void func() {
		cout << "父类func()" << endl;
	}
	void func(int a) {
		cout << "父类func(int a)" << endl;
	}
	void func(float a) {
		cout << "父类func(float a)" << endl;
	}
};
class Son5 : public Base5 {
public:
	void func() {
		cout << "子类func()" << endl;
	}
};
int main() {
	Son5 s;
	
	s.func();

	//屏蔽所有父类中同名函数，重载效果消失
	//s.func(1);
	//s.func(1.05f);

	//父类中的重载函数依然被继承，但是需要加作用域
	s.Base5::func(1);
	s.Base5::func(1.05f);
}
```

## 3.4 子类不能继承的成员

构造函数，析构函数，operator=，private都不能继承。

私有成员不可直接访问，但可以通过public成员间接访问。

## 3.5 多继承

子类可同时继承多个父类。

多继承可能会造成较多歧义，饱受争议。

```c++
class Base1{};
class Base2{};
class Son:extend_type1 Base1, extend_type2 Base2{};
```

多重继承对代码维护性的影响是灾难的， 在设计方法上，任何多继承都可以用单继承代替。

## 3.6 菱形继承

菱形继承：有公共祖先的继承，叫菱形继承。

最底层的子类数据会包含多份公共祖先的数据。

```c++
// 菱形
		Animal
   	  /		   \
    Sheep	   Tuo
	  \        /
       SheepTuo

            
#include<iostream>
#include<string.h>
using namespace std;
class Animal {
public: 
	//通过static可以解决菱形继承数据多份的问题。
	//static int data;
	int data;
};
// 静态成员使用前必须初始化
//int Animal::data = 100;
class Sheep : public Animal{};
class Tuo : public Animal{};

class SheepTuo:public Sheep,public Tuo{};

int main() {
	SheepTuo st;
	// 这个好像必须写，暂时不知道原因
	memset(&st, 0, sizeof(SheepTuo));
	
	// 产生了二义性，从羊那继承了一份data，从驼那继承了一份data，所以有两份data
	// 可以通过static和虚继承来解决。
	//cout<<st.data;

	cout << st.Sheep::data<<endl;
	cout << st.Tuo::data << endl;
	return 0;
}
```

## 3.7 虚继承

虚继承解决 菱形继承中，多份公共祖先数据的问题。

在继承方式前加virtual修饰，子类虚继承父类，子类只会保存一份公共数据。

```c++
#include<iostream>
#include<string.h>
using namespace std;
class Animal1 {
public:
	int data;
};
class Sheep1 : virtual public Animal1 {};
class Tuo1 : virtual public Animal1 {};

class SheepTuo : public Sheep1, public Tuo1 {};

int main() {
	SheepTuo st;
	cout<<st.data;

	return 0;
}
```

# 4 多态

多态性（polymorphism）提供了接口与具体实现之间的另一层隔离，从而将what和how分离开来。

多态性改善了代码的可读性和组织性，同时也使**创建的程序具有可扩展性。**

静态多态（编译时多态，早绑定）：函数重载，运算符重载，重定义

动态多态（运行时多态，晚绑定）：虚函数

**同一类型的多个实例，在执行同一方法时，呈现出多种行为特征——多态**

## 4.1 虚函数

我们在实际场景中，不管一个基类存在多少个子类，都希望有一个父类指针保存某一子类对象的地址，然后通过父类指针去调用子类对象的方法。

理想状态：

你给我一个子类对象（父类指针保存其地址，**运行时绑定**），我就通过这个父类指针区调用这个子类的方法。

不然，你创一个子类，我就要新写一个子类去调用它的方法，这样是低效的。

多态的成立：父类指针（或引用）指向子类的空间。

### 4.1.1 父类指针指向子类对象的问题。

父类指针如果不做任何处理，就直接指向子类对象，那么父类指针访问的仅仅是子类中父类空间的成员，达不到我们的需求。

```c++
#include<iostream>
using namespace std;
class Animal {
public:
	void speak() {
		cout << "我在说话" << endl;
	}
};
class Dog :public Animal {
public:
	void speak() {
		cout << "我在汪汪汪" << endl;
	}
};
int main() {
	Animal* b = new Dog;
	b->speak();//我在说话，我们的需求是调用子类的方法，我在汪汪汪。
	// 父类的指针无法到达子类方法的位置。
    // 我不知道这里怎么去清理堆区空间，这里涉及到虚析构，后面会说到。
	return 0;
}
```

![](./legend/父类指针指向子类空间的问题.png)

### 4.1.2 虚函数定义

成员函数前加virtual修饰。

```c++
#include<iostream>
using namespace std;
class Animal {
public:
	//虚函数
	virtual void speak() {
		cout << "我在说话" << endl;
	}
};
class Dog :public Animal {
public:
	//子类重写父类的虚函数：函数名，返回值类型，参数类型个数顺序，必须完全一致
	void speak() {
		cout << "我在汪汪汪" << endl;
	}
};
class Cat :public Animal {
public:
	//子类重写父类的虚函数：函数名，返回值类型，参数类型个数顺序，必须完全一致
	void speak() {
		cout << "我在喵喵喵" << endl;
	}
};
int main() {
	Animal* b1 = new Dog;
	b1->speak();//我在汪汪汪

	Animal* b2 = new Cat;
	b2->speak();//我在喵喵喵
    // 我不知道这里怎么去清理堆区空间，这里涉及到虚析构，后面会说到。
	return 0;
}
```

### 4.1.3 虚函数动态绑定原理

1. 建立Animal，并且声明speak为虚函数的时候，就会产生一个**虚函数指针vfptr**，虚函数指针指向一张**虚函数表vftable**，这个表记录着这个虚函数的入口地址。
2. Dog继承Animal的时，也会继承虚函数指针和虚函数表， 一旦Dog重写了父类的虚函数，那么此时speak的入口地址就会被替换为Dog::speak。
3. 当我们调用时，用父类指针指向子类的空间，本质上还是调的是父类的speak，但当它去调父类speak的时候，发现这个speak是一个虚函数指针，那么speak的入口地址在哪呢，就需要通过这个指针指向的虚函数表去查，查到入口地址，发现这个入口地址指向的是Dog::speak，所以这里就间接的调用了Dog::speak



![](./legend/虚函数动态绑定原理.png)

```c++
#include<iostream>
using namespace std;

// 接口类
class Animal {
public:
	//虚函数
	virtual void speak() {
		cout << "我在说话" << endl;
	}
};

// 实现类
class Dog :public Animal {
public:
	void speak() {
		cout << "我在汪汪汪" << endl;
	}
};
class Cat :public Animal {
public:
	void speak() {
		cout << "我在喵喵喵" << endl;
	}
};

// 调用类
class Speaker{
public:
	// 动态绑定，接口类指针
	void animalSpeak(Animal* p) {
		p->speak();
	}
};

int main() {
	Speaker sp;
	// 传什么实现类进去，就调用那个实现类的方法。
	sp.animalSpeak(new Dog);
	sp.animalSpeak(new Cat);

	// 我不知道这里怎么去清理堆区空间，这里涉及到虚析构，后面会说到。
	return 0;
}
```

### 4.1.4 重载，重定义，重写

1. 重载：
   - 同一作用域，同名函数，参数的顺序、个数、类型不同 都可以重载。函数重载、运算符重载
   - 函数的返回值类型不能作为重载条件
2. 重定义
   - 有继承，子类 重定义 父类的同名函数（非虚函数）， 参数顺序、个数、类型可以不同。
   - 子类的同名函数会屏蔽父类的所有同名函数（可以通过作用域解决）
3. 重写（覆盖）
   - 有继承，子类 重写 父类的虚函数。
   - 返回值类型、函数名、参数顺序、个数、类型都必须一致。

## 4.2 纯虚函数

如果基类一定会派生子类（基类一般不单独使用），而子类一定会重写父类的虚函数。那么父类的虚函数中的函数体感觉是无意义，可不可以不写父类虚函数的函数体呢？可以的，那就必须了解纯虚函数。

**含有纯虚函数的类，称为抽象类**

```c++
class Animal{
    public:
    	// 注意后面的=0，如果不写就成了函数声明了，而不是纯虚函数
    	virtual void speak(void)=0;
}
```

注意：

1. **一旦类中有纯虚函数，那么这个类 就是抽象类。**
2. 抽象类 不能实例化 对象。（Animal ob；错误）
3. **抽象类 必须被继承 同时 子类 必须重写 父类的所有纯虚函数，否则 子类也是抽象类。**
4. 抽象类主要的目的 是设计 类的接口

```c++
#include<iostream>
using namespace std;

// 接口类
class Animal {
public:
	//纯虚函数
	virtual void speak() = 0;
};

// 实现类
class Dog :public Animal {
public:
	void speak() {
		cout << "我在汪汪汪" << endl;
	}
};
class Cat :public Animal {
public:
	void speak() {
		cout << "我在喵喵喵" << endl;
	}
};

// 调用类
class Speaker {
public:
	//抽象类的主要作用是作为接口。
	void animalSpeak(Animal* p) {
		p->speak();
	}
};

int main() {
	Speaker sp;
	sp.animalSpeak(new Dog);
	sp.animalSpeak(new Cat);

	// error：Animal 无法实例化抽象类
	//Animal a;
	return 0;
}
```

## 4.3 虚析构

virtual修饰析构函数
虚析构：通过父类指针 释放整个子类空间。

和虚函数的动态绑定原理相似，有点区别就是，找到虚析构函数实际指的是子类的析构函数，子类析构之后，会自动调父类的析构函数。

![](./legend/虚析构原理.jpg)

```c++
#include<iostream>
using namespace std;


class Animal {
public:
	//纯虚函数
	virtual void speak() = 0;
	Animal() {
		cout << "animal 构造" << endl;
	}
	// 虚析构函数，通过父类指针释放子类所有空间。
	// 不用虚析构，那么delete *a的时候，只会释放子类中包含的父类部分空间
	virtual ~Animal() {
		cout << "animal 析构" << endl;
	}
};

class Dog :public Animal {
public:
	void speak() {
		cout << "我在汪汪汪" << endl;
	}
	Dog() {
		cout << "Dog 构造" << endl;
	}
	~Dog() {
		cout << "Dog 析构" << endl;
	}
};

int main() {
	Animal* a = new Dog;
	a->speak();
	delete a;

	return 0;
}
```

### 4.3.1 纯虚析构

纯虚析构的本质：是析构函数，各个类的回收工作。而且析构函数不能被继承。

必须为纯虚析构函数提供一个函数体，并且纯虚析构函数 必须在类外实现。

```c++
#include<iostream>
using namespace std;


class Animal01 {
public:
	//纯虚函数
	virtual void speak() = 0;
	//纯虚析构函数，必须在类外实现
	virtual ~Animal01() = 0;
};

class Dog :public Animal01 {
public:
	void speak() {
		cout << "我在汪汪汪" << endl;
	}
	~Dog() {
		cout << "Dog 析构" << endl;
	}
};

Animal01:: ~Animal01() {
	cout << "Animal01 析构" << endl;
}
int main() {
	Animal01* a = new Dog;
	a->speak();
	delete a;

	return 0;
}
```



### 4.3.2 虚析构和纯虚析构的区别

虚析构：virtual修饰，有函数体，不会导致父类为抽象类。

纯虚析构：virtual修饰，=0，函数体必须类外实现，导致父类为抽象类。

# 5 模板

C++泛型编程思想：模板

模板分类：

- 函数模板：在函数中运用模板
- 类模板。

将功能相同，类型不同的函数（类）的类型抽象成虚拟的类型。当调用函数（类实例化对象）的时 候，编译器自动将虚拟的类型 具体化。这个就是函数模板（类模板）。

## 5.1 函数模板



# visual studio

1. [VS2022：如何在一个项目里写多个cpp文件并可以分别独立运行](https://blog.csdn.net/yang2330648064/article/details/123191912)

2. [在VS Studio中管理多个cpp文件或项目](https://blog.csdn.net/Kern5/article/details/127350204)

3. [Visaul Studio不小心点击【从项目中排除】怎么办？](https://zhuanlan.zhihu.com/p/509737464)

4. 快速生成成员函数实现块

   ![](./legend/快速生成成员函数实现框架.png)

5. 无法解析外部符号main

   - 这是说明，cpp文件中找不到入口，main函数

6. 