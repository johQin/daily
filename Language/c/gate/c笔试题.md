# 理论

1. **数组作为参数退化为指针问题**

   ```c
   // 将 数组 作为 函数参数 , 传递时会 退化为高一级的指针 ;如果是多维数组，也只能退化高一级的指针
   // 无论数组是普通数组，还是指针数组都会退化为指针
   
   #include <stdio.h>
   
   void fun(int array[3])			// 退化为int* array， 数组的首地址, 变为指针地址, 函数中无法判定数组的大小;
   {
       printf("fun : sizeof(array)=%d\n", sizeof(array));
   }
   
   int main(int argc, char **args)
   {
       // 将要作为实参的数组
       int array[3] = {1, 2, 3};
       printf("main : sizeof(array)=%d\n", sizeof(array));
   
       // 将数组作为参数传递到函数中
       fun(array);
   
       return 0;
   }
   
   // main : sizeof(array)=12
   // fun : sizeof(array)=8
   
   // 编译器会将 形参中的数组 作为指针处理 , 只会为其分配 指针 所占用的内存;
   // 如果 编译器 将 形参作为 数组处理 , 需要 将数组中的所有元素 , 都要拷贝到栈中 , 如果这个数组很大 , 有几千上万个元素 , 那么该函数的执行效率就很低了 ;
   ```

   - [**不能用二级指针做参数传递二维数组**](https://blog.csdn.net/u011232393/article/details/88298851)

   ```c
   #include <stdio.h>
   
   void fun(int array[][2])		// 退化为int (*array)[2]，只能退化一级，不能完全退化
   {
       printf("fun : sizeof(array)=%d\n", sizeof(array));
   }
   
   int main(int argc, char **args)
   {
       int **ptr;
       fun(ptr);
   
       return 0;
   }
   
   // 多维数组作为函数的形参int array[][y][z]会退化为int (*array)[y][z]
   // 在这里函数形参类型会退化为int (*array)[2]，如果将实参int **ptr赋值给形参int (*array)[2]，会报error: cannot convert 'int**' to 'int (*)[2]'
   ```

   

2. 

# 习题

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
       
       
       std::cout << "----sizeof 字符------" << std::endl;
       char c = 'a';							// 小心, char和'a'在被=动作前, 是两个独立类型, 没关联
       std::cout<< sizeof(c) <<std::endl;      // 结果:1, char类型是1字节
       std::cout<< sizeof('a') <<std::endl;    // C99的标准，    'a'是整型字符常量，常量!常量!常量!被看成是int型， 所以占4字节。
                                               // ISO C++的标准，'a'是字符字面量  ，被看成是char型，所以占1字节。
       
       std::cout << "----sizeof 字符串------" << std::endl;
       std::cout<< sizeof("abc") <<std::endl;  // 4，双引号会在尾部自动添加转义字符'\0',即数据0X00, 所以是4
       // 双引号作用: (1)字符串尾部加0, (2)开辟内存空间, (3)提取地址
       
       std::cout << "----sizeof 指针------" << std::endl;
       char *p="老师,早上好!";
       std::cout<< sizeof(p) <<std::endl;       // 8, 
       std::cout<< sizeof(*p) <<std::endl;       // 1,
       char *a[4];
       std::cout<< sizeof(a)<<std::endl;          // 4 * 8
       std::cout<< sizeof(*(a + 1))<<std::endl;    // 8
       std::cout<< sizeof(**(a + 1))<<std::endl;   // 1
       
       std::cout << "----sizeof 数值------" << std::endl;
       char c=8;
       int  i=32;
       std::cout<< sizeof(c    ) <<std::endl;      // 1, 因为char就是1字节
       std::cout<< sizeof(c+i  ) <<std::endl;      // 4, i是4字节, 运算时c值被隐式转换成int, 运算值是4字节
       std::cout<< sizeof(c=c+i  ) <<std::endl;    // 1, 等同于(c), 编译时, 因为=不被执行, 所以=的右边只是个屁
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

     - 原码从右往左扫描，遇到0直接抄下来，遇到第一个1也直接抄下来，从第一个1往后，所有的数字都取反

       ```
       eg：
       -12
       绝对值源码书写：|-12| = 0000 1100
       从右至左扫描：1111 0100（补码）
       
       法一：
       5的原码：0000 0101
       5的反码：1111 1010 （原码取反）
       5的补码：1111 1011 （反码 +1）
       
       法二：
       5的原码：0000 0101
       5的补码：1111 1011（原码从右往左扫描，遇到0直接抄下来，遇到第一个1也直接抄下来，从第一个1往后，所有的数字都取反）
       ```
       
     - 八位二进制数表示有符号数,为什么最大值为127而不是128：
     
       - 八位二进制数,共有 256 个编码,只能表示 256 个有符号数。一半(128个)是负数,一半(128个)是零和正数。正数的最大值,如果是 128,那就需要 129 个编码
     
       - ```
         127:	0111 1111
         -128:	1000 0000
         -127:	1000 0001
         ```
     
     - **最值溢出**：类型的最大数 + 1 = 类型的最小数，类型的最小数 - 1 = 类型的最大数
     
       ```c
           signed char i;
           i=-129;
           printf("%d\n",i);	// 127，-127 = 类型的最小值（-128） - 1
           i=129;
           printf("%d\n",i);	// -127，-128 = 类型的最大值（127） + 1，-128 + 1 = -127
       
       
       
       ```
     
     - signed short最大值：32767，最小值：-32768

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

6. **函数 fun 的声明为 int fun(int *p[4]), 以下哪个变量可以作为fun的合法参数（）**

   - **n 维数组名称本质 是 n-1 维数组 的指针**
   - **`type** (**p)[]`，type 类型后面描述的是数组存储的类型，括号里面是指针的级别**
   - **将 数组 作为 函数参数 , 传递时会 退化为高一级的指针 ;如果是多维数组，也只能退化高一级的指针**

   ```c
   A int a[4][4];				// a 是 二级指针
   B int **a;					// a 是 二级指针
   C int **a[4];				// a 是 三级指针
   D int (*a)[4];				// a 是 一级指针
   
   // 数组作为函数形参会退化为指针，所以这里的int fun(int *p[4])会退化为int fun(int** p)
   
   // int a[4][4]，a本质上是一个一维数组包含4个元素的指针int (*p)[4]
   // 由于函数参数退化为int **p， 所以这里的类型转化将失败，报 error: cannot convert 'int (*)[2]' to 'int**'
   
   // int** a[4]; a 指向数组，数组里面存放的是二级指针, a 可以看做三级指针
   
   // int (**a)[4]; a 是一个二级指针，数组里面存放 int 型，error: cannot convert 'int (**)[2]' to 'int**'
   
   // int (*a)[4]; a 是个一级指针，数组里面存放 int 型, int(*a)[4]和 int a[4]的区别在于一个是指针，一个是数组名(指针常量)
   // error: cannot convert 'int (*)[2]' to 'int**'
   
   ```

7. 