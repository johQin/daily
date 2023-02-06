# C++

C++融合了3种编程模式：面向过程编程（**Process-oriented programming，POP**），面向对象编程（**OOP**），泛型编程.

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
   //包含了目前c++所包含的所有头文件！！！！
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

1. 注释

   - 单行注释：**`//`**
   - 多行注释：**`/**/`**

2. 变量

   - 作用：分配一段内存并做标识，方便操作它
   - 声明：**`数据类型 变量名`**

3. 常量

   - 符号常量：`#define DAY 7`，也称宏常量，这个定义语句后面不能加**" ; "**，否则会报语法错误

   - 常变量：`const int MONTH = 12;`

   - 常量不可以被修改，实型常量被默认为是一个double类型的常量

   - ```c++
     #include<iostream>
     #include <bits/stdc++.h>
     using namespace std;
     //常量的定义方式
     //1.符号常量（宏常量）
     // 定义语句后面不能加“;”
     #define Day 7
     int main() {
     	cout << "你好，一周总共=" << Day << "天" << endl;
     	
     	// 2 const 修饰的常变量
     	const int month = 12;
     	cout << "one year have " << month << " month" << endl;
         return 0;
     }
     ```

4. 关键字

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



# 1 数据类型

1. 基本数据类型
   - 整型类型
     - 整型（int，2个or4个字节），无符号整型（unsigned int）
     - 短整型（short，2个字节），无符号短整型（unsigned short）
     - 长整型（long，4个字节），双长整型（long long，8个字节），无符号...
     - 字符型（char，1个字节），有符号（)，无符号（0~2^8）
     - 布尔型（bool，占用1个字节）
   - 浮点类型（以小数或指数形式出现）
     - 单精度浮点型（float，4个字节）
     - 双精度浮点型（double，8个字节），长双精度（long double，8or 16个字节）
     - 复数浮点型（float_complex，double_complex， long long_comple)
2. 复合数据类型
3. 构造数据类型

```c++
#include<iostream>
using namespace std;
#include<bits/stdc++.h>
int main() {

	// 1.整型数据
	short num1 = 10;
	int num2 = 10;
	//unsigned num2 = 10;
	//unsigned int num2 = -10; //会输出它的补码
	long num3 = 10;
	long long num4 = 10;
	cout << "num1 = " << num1 << endl;
	cout << "num2 = " << num2 << endl;
	cout << "num3 = " << num3 << endl;
	cout << "num4 = " << num4 << endl;

	//2.利用sizeof计算数据类型或变量所占的空间大小
	//sizeof(type or var)，单位：字节
	cout << "short所占的空间：" << sizeof(short) << endl;
	cout << "int所占的空间：" << sizeof(int) << endl;
	cout << "short num1所占的空间：" << sizeof(num1) << endl;

	//3.实型
	//默认情况下，输出一个小数，会显示出6位有效数字
	float f1 = 3.14f;
	double d1 = 3.14;
	cout << "f1= " << f1 << endl;
	cout << "d1= " << d1 << endl;
	cout << "float所占空间" << sizeof(float) << endl;
	cout << "double所占空间" << sizeof(double) << endl;
	float f2 = 3e2; //3*10^2
	float f3 = 3e-2;
	cout << "f2= " << f2 << endl;
	cout << "f3= " << f3 << endl;

	//4.字符型
	//只占一个字节，字符型变量并不是把字符本身放到内存中存储，而是将对应的ASCII编码放入到存储单元
	char ch1 = 'a';
	cout << "ch1= " << ch1 << endl;
	cout << "char所占空间 " << sizeof(char) << endl;
	// 字符型变量的值只能包含一个字符，且只能用单引号，赋值不能用双引号，"b"
	cout << "ch1='a'对应的ASCII码值" << (int)ch1 << endl;

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

# 2 运算符

1. 算术运算符，`+ - * / % ++ --`
2. 关系运算符，`> < == >= <= !=`
3. 逻辑运算符，`! && || `
4. 位运算符，`<< >> ~ | ^ &`
5. 赋值运算符，`= += -=等`
6. 条件运算符（三目运算符），`cond ? t_exp : f_exp`
7. 逗号运算符
8. 指针运算符，`* &`
9. 字节数运算符，`sizeof`
10. 强制类型转换符，`(type_name) (exp)`
11. 成员运算符， `.->`

```c++
#include<iostream>

using namespace std;
int main() {
	int a1 = 10;
	int b1 = 3;
	// 两个整数相除，结果依然是一个整数，会直接去掉小数部分（不会四舍五入）
	cout << a1 / b1 << endl;

	double a2 = 3.14;
	double b2 = 0.1;
	// 两个小数不能进行取余运算
	//cout << a2 % b2 < endl;
}
```

# 3 程序结构

## 3.1 选择结构

### 3.1.1 if结构

```c++
// 1 单判断，没有else
if(expr) 语句1;

// 2 双判断
if(expr) 语句1;
else 语句2;

// 3 多判断
if(expr1) 语句1;
else if(expr2) 语句2;
else if(expr3) 语句3;
...
else 语句n;
```

注意：

1. 如果语句是复合语句（多语句）需要用花括号括起来。
2. if语句可以嵌套

### 3.1.2 switch结构

```c
switch(expr){
    case constant1: 语句1;break;
    case constant2: 语句2;break;
    ...
    case constantn: 语句n;break;
    default: 语句n+1;
}
```

注意：

1. expr的值应为整数类型（包括字符型）
2. 每个case的常量必须互不相同
3. 可以没有default语句
4. case后面如果有多个语句可以不用花括号，因为通过case找到语句执行的入口后，后面会顺序执行。所以如果不想入口后面的case执行，就必须添加break
5. 多个case可以共用一个语句

## 3.2 循环结构

### 3.2.1 while结构

```c
// while，当循环条件为真，就执行循环体;
// 先判断，后执行
int i = 1, sum = 0;
while(i <= 100) {
    sum = sum + i;
    i++;
}
// do...while
// 先执行，判断
int i = 1, sum = 0;
do{
    sum = sum + i;
    i++;
}while(i <= 100)
```

### 3.2.2 for结构

```c
for(expr1; expr2; expr3) {
    //expr1：设置初始条件，只执行一次，可以为0个，但其后的省略号不能省，可以为循环言变量设置初值，也可以是与循环变量无关的表达式
    //expr2: 循环条件表达式，先判断，后执行
    //expr3：循环调整，执行完循环体后执行
    
    //expr1和expr3可以是一个简单的表达式，也可以是逗号分割多个简单表达式。
}
```

## 3.3 跳转语句

1. break：提前终止循环

2. continue：提前结束本次循环

3. goto：改变程序执行的顺序，从goto执行处跳转至标记处

   - C 语言中的 **goto** 语句允许把控制无条件转移到**同一函数内**的被标记的语句。

   - 标签在哪里，goto语句就可以往哪跳，可往前跳，可往后跳

   - 应用场景：终止程序在某些深度嵌套的结构的处理过程

   - 在程序中不建议使用goto语句，以免造成程序流程混乱

   - ```c++
     for(...)
         for(...)
        {
             for(...)
            {
                 if(disaster)
                     goto error;
            }
        }
         …
     error:
      if(disaster)
     ```

   - 

4. 



# visual studio

1. [VS2022：如何在一个项目里写多个cpp文件并可以分别独立运行](https://blog.csdn.net/yang2330648064/article/details/123191912)
2. [在VS Studio中管理多个cpp文件或项目](https://blog.csdn.net/Kern5/article/details/127350204)
3. [Visaul Studio不小心点击【从项目中排除】怎么办？](https://zhuanlan.zhihu.com/p/509737464)