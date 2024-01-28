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
