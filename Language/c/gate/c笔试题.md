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

   ![](../legend/笔试abc常量区.png)

2. sizeof()

   - 函数的实参若为数组，那么传址
   - sizeof(x)在**编译**时确定其值, 计算出x在内存中所占字节数。所以, **括号内的赋值和函数, 不会被执行，但括号内的运算还是会执行**

   ```c++
   void func(short b[])		// 可以看作是void func(short* b)
   {
   	cout << sizeof(b)<<endl; //一个地址所占字节数为4byte(32位电脑)， 8byte（64位电脑）
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
   
   	
   
   	std::cout << "----sizeof 数组------" << std::endl;
   	short c[2][3] = { {1,2,3},{4,5,6} };
   	short c[2][3] = { {1,2,3},{4,5,6} };
       std::cout << sizeof(c) << std::endl;	//12，整个二维数组的大小
       std::cout << sizeof(c + 1) << std::endl; //8，c+1 指向的是c[1]，所以c+1是一个地址值，而c[1]指向的是c[1][0]
       std::cout << sizeof(*(c + 1)) << std::endl;//6，c+1取星后，获得c[1],而c[1]是一个一维数组名，所以是2x3
       // 二维数组c -> c[0]，而c[0]->c[0][0]
       
       std::cout << "----sizeof 函数------" << std::endl;
       int Sum(int i, short s);
       std::cout<< sizeof(Sum(32, 8)) <<std::endl;		//// 结果:4, 只会判断返回类型的大小. 函数是不会执行的
       
       // 函数传址
       short b[100];
   	func(b);//数组名作为实参，是传址
       
       
   	std::cout << "----sizeof 结构体------" << std::endl;
       struct T1{
           int   a;                  // 成员随意位置
           char  b;
           int   c;
           short d;
       }t1;
       std::cout<< sizeof(t1) <<std::endl;     // 结果:16, 4+4+4+4
       struct T2{
           int   a;                  // 合理安排成员位置
           char  b;
           short d;
           int   c;
       }t2;
       std::cout<< sizeof(t2) <<std::endl;     // 结果:12, 4+4+4, 设计结构时，调整成员的位置，可节省存储空间。
       // 字节对齐,为快速处理数据,内存是按32位读取写的,而不是一字节一字节地读写
       // 结构体的首地址自动对齐至能被对齐字节数大小所整除。
       // 结构体每个成员在结构体内的偏移地址都是成员大小的整数倍，否则, 在前方填充byte。
       // 结构体的总大小为结构体对齐字节数大小的整数倍
   }
   ```

3. strlen遇0不再计数

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

   

4. [复数以补码的形式存储](https://blog.csdn.net/b1480521874/article/details/102723491)

   - 补码：

     - 绝对值源码二进制书写，

     - 从右往左扫描，遇到0直接抄下来，遇到第一个1也直接抄下来，从第一个1往后，所有的数字都取反

       ```
       eg：
       -12
       绝对值源码书写：|-12| = 0000 1100
       从右至左扫描：1111 0100（补码）
       ```

       

5. 指针指向

   - 取星*元素维度降一维，取址&元素维度加一维

   ![](../legend/指针指向问题.png)

   ```c
   	int a[2][2][3] = { {{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}} };
   
   	int* ptr = (int*)(&a + 1);
   	// &a处在第四维，元素的跨度为12
   
   	printf(" %d %d", *(int*)(a + 1), *(ptr - 1));//7 12
   	// a处在第三维，元素的跨度为6，故a + 1指向 7
   
   	printf(" %d %d", *(int*)(*a + 1), *(int*)(*a + 2));//4 7
   	// *a处在第二维，元素的跨度为3, 故*a + 1指向4，*a + 2 直向7
   
   	return 0;
   ```

6. 