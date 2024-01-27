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
```

