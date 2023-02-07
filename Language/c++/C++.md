# C++

C++融合了3种编程模式：面向过程编程（**Process-oriented programming，POP**），面向对象编程（**OOP**），泛型编程。

C++中面向过程的语法兼容大部分C语言，只是在版本不同时，有兼容差异，在一定程度上可以说C++是C语言的超集

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

解决命名冲突，用于约束标识符name的作用范围，name可以包括常量，变量，函数，结构体，枚举，类，对象等等

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



# visual studio

1. [VS2022：如何在一个项目里写多个cpp文件并可以分别独立运行](https://blog.csdn.net/yang2330648064/article/details/123191912)
2. [在VS Studio中管理多个cpp文件或项目](https://blog.csdn.net/Kern5/article/details/127350204)
3. [Visaul Studio不小心点击【从项目中排除】怎么办？](https://zhuanlan.zhihu.com/p/509737464)