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

	//6.bool
	bool flag1 = true;
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



# visual studio

1. [VS2022：如何在一个项目里写多个cpp文件并可以分别独立运行](https://blog.csdn.net/yang2330648064/article/details/123191912)
2. [在VS Studio中管理多个cpp文件或项目](https://blog.csdn.net/Kern5/article/details/127350204)
3. [Visaul Studio不小心点击【从项目中排除】怎么办？](https://zhuanlan.zhihu.com/p/509737464)