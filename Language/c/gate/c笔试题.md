1. ["abc" 在常量区还是栈区](https://blog.csdn.net/qq_40024275/article/details/100526940)

   ```c
   	void func() {
   		char a[] = "abc"; //栈
   		char b[] = "abc"; //栈
   		char* c = "abc"; //abc在常量区，c在栈上。
   		char* d = "abc"; //编译器可能会将它与c所指向的"abc"优化成一个地方。
   		const char e[] = "abc"; //栈
   		const char f[] = "abc"; //栈
   
   		cout << a << "/" << b << "/" << c << "/" << d << "/" << e << "/" << f << endl;
           	// abc/abc/abc/abc/abc/abc
   
   		cout << *a << *b << *c  << *d << *e << *f << endl;//aaaaaa
   
   		cout << strcmp(a,b) << strcmp(c, d) << strcmp(e, f) << endl;// 000，说明字符串是相等的
   
   		// 比较的是地址值,看是否指向同一个地址
   		cout << (a == b ? 1 : 0)  << (c == d ? 1 : 0)  << (e == f ? 1 : 0) << endl; //010
   
   	}
   ```

   ![](D:/gitRepository/daily/Language/c/legend/笔试abc常量区.png)

2. sizeof()

   - 函数的实参若为数组，那么传址

   ```c++
   void func(short b[100])
   {
   	cout << sizeof(b)<<endl; //4，一个地址所占字节数为4byte
   }
   int main() {
   	cout << sizeof(char) << endl;//1
   	cout << sizeof(short) << endl;// 2
   	cout << sizeof(int) << endl;// 4
   	cout << sizeof(long int) << endl;//4
   	cout << sizeof(double) << endl;//8
   	cout << "------------" << endl;
   
   	short a[10] = {0};
   	cout<<sizeof(a)<<endl;//20
   	cout << sizeof(*a) << endl;// 2，*a 等价于a[0]
   	cout << a[110] << endl;// 0，这里索引早已越界，所以读出的值，是不确定的
   	cout<< sizeof(a[110])<<endl;//2，但所占大小依旧是2byte
   
   	
   	cout<< &a <<endl; //0079FBEC, *a 等价于a[0]
   	cout << &a[0] << endl; //0079FBEC，，&a等价于&a[0]
   	cout << sizeof(&a) << endl;//4，地址所占的空间
   	cout<< sizeof(&a[0]) <<endl;//4，地址所占的空间
   
   	short b[100];
   	func(b);//数组名作为实参，是传址
   
   	cout << "----------" << endl;
   	short c[2][3] = { {1,2,3},{4,5,6} };
   	cout << sizeof(c) << endl;	//12，整个二维数组的大小
   	cout << sizeof(c + 1) << endl; //4，c+1 指向的是c[1]，所以c+1是一个地址值，而c[1]指向的是c[1][0]
   	cout << sizeof(*(c + 1)) << endl;//6，c+1取星后，获得c[1],而c[1]是一个一维数组名，所以是2x3
   	//二维数组c -> c[0]，而c[0]->c[0][0]
   }
   ```

3. strlen遇0不再计数

   - [复数以补码的形式存储](https://blog.csdn.net/b1480521874/article/details/102723491)，补码=反码（绝对值原码取反）+ 1

   ```c++
   	char a[1000];
   	int i;
   	for (i = 0; i < 1000; i++)
   	{
   		a[i] = -1 - i;
   		printf("%d", a[i]);
   		// -1~-128 127~1 0 -1....
   	}
   	printf("%d", strlen(a));//255，strlen遇0就不再计数。
   ```

   

4. 