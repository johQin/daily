# pattern design

设计模式是对软件设计中普遍存在（反复出现）的各种问题，所提出的解决方案。由埃里希·伽玛等人在1990年从建筑设计领域引入计算机科学

设计模式在扩展性、复用性、稳定性，站在大工程的角度看待问题。有时候我们会感觉没有必要，但是越往后越感觉很受益。

它不是针对于某个功能如何实现，而是在工程结构怎么更合理

面向对象=>功能模块（设计模式+数据结构算法）=>框架（多种设计模式）=>架构

使用过什么设计模式，怎么使用的，解决了什么问题

7设计原则和23中设计模式

应用场景->设计模式->剖析原理->分析实现步骤->代码实现->框架或项目源码分析

# 1 七大原则

程序员面临着来自

- 高内聚性、低耦合性
- 维护性
- 扩展性：当需要添加新功能时，方便，开发成本低
- 重用性：相同功能的代码，不用多次编写
- 可读性：编程规范性，程序易于阅读
- 可靠性：增加新功能后，对原来的功能没有影响
- 灵活性

等多方面的挑战。设计模式就是用来解决这些问题。

七大原则作为设计模式的基础

含有的七大原则：

1. 单一职责原则
2. 接口隔离原则
3. 依赖倒转原则
4. 里氏替换原则
5. 开闭原则ocp
6. 迪米特法则
7. 合成复用原则

谐音记忆：丹姐依你，开底盒

## 1.1 单一职责原则

Single Responsibility Principle

对类来说的，即一个类应该只负责一项职责。如类A 负责两个不同职责：职责1，职责2。当职责1 需求变更而改变A 时，可能造成职责2 执行错误，所以需要将类A 的**粒度**分解为A1，A2。

粒度：**类级别和方法级别**

### 1.1.1 注意事项和细节

1. 降低类的复杂度，一个类只负责一项职责。
2. 提高类的可读性，可维护性
3. 降低变更引起的风险
4. 通常情况下，我们应当遵守单一职责原则，只有逻辑足够简单，才可以在代码级违反单一职责原则；**只有类中方法数量足够少，可以在方法级别保持单一职责原则**

## 1.2 接口隔离原则

Interface Segregation Principle

客户端不应该依赖它不需要的接口，即一个类对另一个类的依赖应该建立在最小的接口上。

下面的类图不符合接口隔离原则

![](legend/segregation_false.png)

### 1.2.1 分析和解决

1. 类A 通过接口Interface1 依赖类B，类C 通过接口Interface1 依赖类D，如果接口Interface1 对于类A 和类C来说不是最小接口，那么类B 和类D 必须去实现他们不需要的方法
2. 将接口Interface1 拆分为独立的几个接口，类A 和类C 分别与他们需要的接口建立依赖关系。也就是采用接口隔离原则
3. 接口Interface1 中出现的方法，根据实际情况拆分为三个接口
4. 调整为符合接口隔离原则

![](legend/segregation_true.jpg)

## 1.3 依赖倒转原则

普通的思维：高层应依赖低层。

依赖倒转：低层应依赖高层。

### 1.3.1 基本介绍

依赖倒转原则(Dependence Inversion Principle)是指：

1. 高层模块不应该依赖低层模块，二者都应该依赖其**抽象**（抽象类、接口，不要依赖具体的子类）
2. 抽象不应该依赖**细节**（实现类），细节应该依赖抽象
3. **依赖倒转(倒置)的中心思想是面向接口编程**
4. 依赖倒转原则是基于这样的设计理念：相对于细节的多变性，抽象的东西要稳定的多。
   - 以抽象为基础搭建的架构比以细节为基础的架构要稳定的多。
   - 在java 中，抽象指的是接口或抽象类，细节就是具体的实现类。
5. 使用接口或抽象类的目的是制定好规范，而不涉及任何具体的操作，把展现细节的任务交给他们的实现类去完成

```java
public class Inversion{
    public static void main(String[] args) {
        Person person=new Person();
        person.receive(new Email());
        person.receive(new Wechat());
    }
}

/**
 * 完成Person接收消息的功能
 * 方式1：
 * 1. 传给receive一个Eamil对象，通过调用Email的getInfo方法，返回邮件消息。简单，比较容易想到,
 * 2. 如果我们获取的对象是微信，短信等等，则新增类，同时Person也要增加相应的接收方法
 * 3. 解决思路：引入一个抽象的接口IRceiver，表示接受者，这样Person类与接口IReceiver发生依赖
 * 4. 因为Email，weixin等等属于接收的范围，他们各自实现IReceiver 接口就ok，这样我们就符合依赖倒转原则
 */
class Person{
    public void receive(IReceiver iReceiver){
        System.out.println(iReceiver.getInfo());
    }
}
interface IReceiver{
    public String getInfo();
}
class Email implements IReceiver{
    public String getInfo(){
        return "邮件消息：hello Email";
    }
}
class Wechat implements IReceiver{
    public String getInfo(){
        return "微信消息：hello Wechat";
    }
}
```



### 1.3.2 依赖关系的传递

1. 接口传递
2. 构造方法传递
3. setter 方式传递

### 1.3.3 注意事项和细节

1. 低层模块尽量都要有抽象类或接口，或者两者都有，程序稳定性更好
2. 变量的声明类型尽量是抽象类或接口, 这样我们的变量引用和实际对象间，就存在一个缓冲层，利于程序扩展和优化
3. 继承时遵循里氏替换原则

## 1.4 里氏替换原则

### 1.4.1 OO 中的继承性的思考和说明

1. 继承包含这样一层含义：**父类中凡是已经实现好的方法，实际上是在设定规范和契约**，虽然它不强制要求所有的子类必须遵循这些契约，但是如果子类对这些已经实现的方法任意修改，就会对整个继承体系造成破坏。
2. 继承在给程序设计带来便利的同时，也带来了弊端。比如使用继承会给程序带来侵入性，程序的可移植性降低，增加对象间的耦合性，如果一个类被其他的类所继承，则当这个类需要修改时，必须考虑到所有的子类，并且父类修改后，所有涉及到子类的功能都有可能产生故障
3. 问题提出：在编程中，如何正确的使用继承? => 里氏替换原则

### 1.4.2 基本介绍

1. 所有引用基类的地方必须能透明地使用其子类的对象（替换）。
2. 在使用继承时，遵循里氏替换原则，**在子类中尽量不要重写父类的方法**。
3. 里氏替换原则告诉我们，继承实际上让两个类耦合性增强了，在适当的情况下，**可以通过聚合，组合，依赖来解决问题。**

## 1.5 开闭原则

### 1.5.1 基本介绍
1. 开闭原则（Open Closed Principle）是编程中最基础、最重要的设计原则
2. 一个软件实体如类，**模块和函数应该对扩展开放(对提供方)，对修改关闭(对使用方)。用抽象构建框架，用实现扩展细节。**
3. **当软件需要变化时，尽量通过扩展软件实体的行为来实现变化，而不是通过修改已有的代码来实现变化。**
4. 编程中遵循其它原则，以及使用设计模式的目的就是遵循开闭原则。

```java
public class OCPimprove{
    public static void main(String[] args) {
        GraphEditor ge=new GraphEditor();
        ge.drawShape(new Ractangle());
        ge.drawShape(new Circle());
        ge.drawShape(new Triangle());
    }
}
//增加新功能，提供方增加了新的实现类，使用方没有任何改变
//使用方
class GraphEditor{
    public void drawShape(Shape s){
        s.draw();
    }
}

//功能提供方
abstract class Shape{
    public draw();
}
class Ractangle extends Shape{
    @Override
    public draw(){
       System.out.println("绘制矩形")
    }
}
class Circle extends Shape{
    @Override
    public draw(){
        System.out.println("绘制圆形")
     }
}
//新增三角形
class Triangle extends Shape{
    @Override
    public draw(){
        System.out.println("绘制三角形")
     }
}
```



## 1.6 迪米特原则

Demeter Principle

不要越层管理，越级联系

### 1.6.1 基本介绍

1. 一个对象应该对其他对象保持最少的了解
2. 类与类关系越密切，耦合度越大
3. 迪米特法则(Demeter Principle)又叫**最少知道原则**，即一个类对自己依赖的类知道的越少越好。也就是说，对于被依赖的类不管多么复杂，都尽量将逻辑封装在类的内部。对外除了提供的public 方法，不对外泄露任何信息
4. 迪米特法则还有个更简单的定义：**只与直接的朋友通信**
5. 直接的朋友：每个对象都会与其他对象有耦合关系，只要两个对象之间有耦合关系，我们就说这两个对象之间是朋友关系。耦合的方式很多，依赖，关联，组合，聚合等。其中，我们称出现成员变量，方法参数，方法返回值中的类为直接的朋友，而出现在局部变量中的类不是直接的朋友。也就是说，**陌生的类最好不要以局部变量的形式出现在类的内部。**

### 1.6.2 注意事项和细节

1. 迪米特法则的核心是降低类之间的耦合
2. 但是注意：由于每个类都减少了不必要的依赖，因此迪米特法则只是要求降低类间(对象间)耦合关系， 并不是要求完全没有依赖关系

## 1.7 合成复用原则

原则尽量使用合成（组合：手臂和人）/聚合（人和俱乐部）的方式，而不是使用继承

![](legend/combine_together.jpg)

## 1.8 设计原则的核心思想

1. 找出应用中可能需要变化之处，把它们独立出来，不要和那些不需要变化的代码混在一起。
2. 针对接口编程，而不是针对实现编程。
3. 为了交互对象之间的松耦合设计而努力。

# 2 UML

**面向对象软件开发需要经过OOA(面向对象分析)，OOD(面向对象设计)和OOP(面向对象编程)三个阶段**

OOA对目标系统进行分析，建立分析模型，并将之文档化。

OOD用面向对象的思想对OOA的结果进行细化，得出设计模型。

OOA和OOD的分析、设计结果需要统一的符号来描述、交流并记录。

UML就是这种用于描述、记录OOA和OOD结果的符号表示法。

UML2.0将图分为静态图和动态图，一共包括13中正式图形。最常用的UML图包括：用例图、类图、组件图、部署图、顺序图、活动图和状态机图

## 2.1 [类图](<https://plantuml.com/zh/class-diagram>)

类的静态内部结构在类图上使用包含三个部分的矩形（类名，属性，方法）来描述。类图除可以表示实体的静态内部结构之外，还可以表示实体之间的相互关系。

![](E:/note/pattern/legend/basic_inner.png)

实体间的相互关系包括

- 关联（聚合，组合）
  - 聚合，一对多关的关系，像篮球俱乐部，由多名学员聚合而成，但学员还可以同时是其他实体的一部分。用空心菱形的实线表示
  - 组合，一对一关的关系，像人体，由手臂等部分组合而成，用实心菱形的实线表示
  - 双向关联（实线）、多重性关联
- 泛化（与继承同一个概念）
  - 子类是特殊的父类，继承关系用带实现的空心三角形表示，由子类指向父类
- 依赖
  - 如果一个类的改变会导致另一个类的改动，则称这两个类之间存在依赖。
  - 依赖关系使用带箭头的虚线表示，其中箭头指向被依赖的实体。
- 实现
  - 接口与实现类，用虚线加空心三角表示。

![](E:/note/pattern/legend/basic_class.png)

## 2.2 [Vscode plantUML](<https://blog.csdn.net/qq_26819733/article/details/84895850>)

1. 安装辅助软件：Graphviz，要默认配置环境变量
2. 在Vscode中安装插件：PlantUML
3. 新建.uml文件
4. 预览.uml的快捷键：【 Alt + D 】

# 3 设计模式

1. 设计模式是程序员在面对同类软件工程设计问题所总结出来的有用的经验，模式不是代码，而是某类问题的通用解决方案，设计模式（Design pattern）代表了最佳的实践。这些解决方案是众多软件开发人员经过相当长的一段时间的试验和错误总结出来的。
2. 设计模式的本质提高软件的维护性，通用性和扩展性，并降低软件的复杂度。

## 3.1 设计模式分类

1. 创建型模式（5个）

   - 单例模式
   - 工厂模式

   - 抽象工厂模式

   - 原型模式

   - 建造者模式

2. 结构型模式（7个）

   - 适配器模式
   - 桥接模式
   - 装饰模式
   - 组合模式
   - 外观模式
   - 享元模式
   - 代理模式

3. 行为型模式（11个）

   - 模板方法模式
   - 命令模式
   - 访问者模式
   - 迭代器模式
   - 观察者模式
   - 中介者模式
   - 备忘录模式
   - 解释器模式
   - 状态模式
   - 策略模式
   - 职责链模式

# 4 单例模式

所谓类的单例设计模式，就是采取一定的方法保证在整个的软件系统中，对某个类只能存在一个对象实例，并且该类只提供一个取得其对象实例的方法(静态方法)。

例如：Hibernate 的SessionFactory，它充当数据存储源的代理，并负责创建Session 对象。SessionFactory 并不是轻量级的，一般情况下，一个项目通常只需要一个SessionFactory 就够，这是就会使用到单例模式。

## 4.1 八种方式

1. 饿汉式(静态常量)
2. 饿汉式（静态代码块）
3. 懒汉式(线程不安全)
4. 懒汉式(线程安全，同步方法)
5. 懒汉式(线程安全，同步代码块)
6. 双重检查
7. 静态内部类
8. 枚举

名词解释：

- 饿汉式：不管程序是否需要这个对象的实例，总是在类加载的时候就先创建好实例，理解起来就像不管一个人想不想吃东西都把吃的先买好，如同饿怕了一样。
- 懒汉式：如果一个对象使用频率不高，占用内存还特别大，明显就不合适用饿汉式了，这时就需要一种懒加载的思想，当程序需要这个实例的时候才去创建对象，就如同一个人懒的饿到不行了才去吃东西。

## 4.2 饿汉式(静态常量)

### 4.2.1 步骤

1. 构造器私有化(防止new)
2. 类的内部创建对象
3. 向外暴露一个静态的公共方法。getInstance

```java
public class Singleton{
    //1.构造器私有化
    private Singleton(){}
    //2.类的内部创建对象
    private static final Singleton instance=new Singleton();
    //3.向外暴露静态公共方法
    public static Singleton getInstance(){
        return instance;
    }
}
```

### 4.2.2 优缺点

1. 优点：这种写法比较简单，就是在类装载的时候就完成实例化。**避免了线程同步问题。**
2. 缺点：在类装载的时候就完成实例化，**没有达到Lazy Loading** 的效果。如果从始至终从未使用过这个实例，则会造成内存的浪费
3. 结论：这种单例模式可用，可能造成内存浪费

## 4.3 饿汉式（静态代码块）

```java
public class Singleton{
    //1.构造器私有化
    private Singleton(){}
    private static Singleton instance;
    //2.静态初始化块
    static {
        instance=new Singleton();
    }
    //3.向外暴露静态公共方法
    public static Singleton getInstance(){
        return instance;
    }
}
```

优缺点如上

## 4.4 懒汉式（线程不安全）

```java
public class Singleton{
    private Singleton(){}
    private static Singleton instance;
    //用到的时候创建
    public Singleton getInstance(){
        if(instance==null){
            instance=new Singleton();
        }
        return instance;
    }
}
```

### 优缺点

1. 优点：起到了Lazy Loading 的效果，但是只能在单线程下使用。
2. 缺点：如果在多线程下，一个线程进入了if (singleton == null)判断语句块，还未来得及往下执行，另一个线程也通过了这个判断语句，这时便会产生多个实例。所以在多线程环境下不可使用这种方式
3. 结论：在实际开发中，不要使用这种方式.

## 4.5 懒汉式（线程安全，synchronized）

```java
public class Singleton{
    private Singleton(){}
    private static Singleton instance;
    //用到的时候创建
    //加了线程同步排队synchronized
    public static synchronized Singleton getInstance(){
        if(instance==null){
            instance=new Singleton();
        }
        return instance;
    }
}
```

### 优缺点

1. 优点：解决了线程安全问题
2. 缺点：效率太低了，每个线程在想获得类的实例时候，执行getInstance()方法都要进行同步。而其实这个方法只执行一次实例化代码就够了，后面的想获得该类实例，直接return 就行了。方法进行**同步效率太低**
3. 结论：在实际开发中，不推荐使用这种方式

## 4.6 懒汉式（线程安全，同步代码块）

```java
public class Singleton{
    private Singleton(){}
    private static Singleton instance;
    //用到的时候创建
    public static Singleton getInstance(){
        if(instance==null){
            
            //无意义
            synchronized{
                instance=new Singleton();
            }
            
        }
        return instance;
    }
}
```

这种方式，本意是想对第四种实现方式的改进，因为前面同步方法效率太低，改为同步产生实例化的代码

但是这种同步并不能起到线程同步的作用。加入一个线程进入if（singleton==null）判断语句块，还未来得及往下执行，另一线程也通过了这个判断语句，这是便会产生多个实例

结论：**在实际开发中，不能使用这种方式。**

## 4.7 双重检查

```java
public class Singleton{
    private Singleton(){}
    //加volatile，线程共享变量值，有改变时立马，写入主存，在一定程度上有线程同步的效果
    private static volatile Singleton instance;
    //用到的时候创建
    public static Singleton getInstance(){
        if(instance==null){
            //加了线程同步排队synchronized
            synchronized(Singleton.class){
                if(instance==null){
                    instance=new Singleton();
                }
            }
        }
        //当第一次多线程A，B（或者更多）都通过了第一重检查，都进入排队阶段。
        //A先进入synchronized代码块，执行代码内容，使instance实例化。
        //B（或者更多）在排队等候后，同样再判断，但不执行实例化。
        //但是在之后线程访问中，在第一重检查就会拦下，直接执行返回
        //因为instance已经实例化，就不会再进入synchronized代码块，无需再执行排队等待操作。
        //效率提高
        return instance;
    }
}
```

### 优点

1. Double-Check 概念是多线程开发中常使用到的，如代码中所示，我们进行了两次if (singleton == null)检查，这样就可以保证线程安全了。
2. 这样，实例化代码只用执行一次，后面再次访问时，判断if (singleton == null)，直接return 实例化对象，也避免的反复进行方法同步.
3. 线程安全；延迟加载；效率较高
4. 结论：在实际开发中，推荐使用这种单例设计模式

## 4.8 静态内部类

```java
public class Singleton{
    private Singleton(){}
    //1.在Singleton执行类装载的时候，内部类SingletonInstance的内部并不会执行类加载，从而实现懒加载
    private static class SingletonInstance{
        //这里给了一个final，类装载后，不再执行赋值操作，在一定程度上增强了单例
        private static final Singleton INSTANCE=new Singleton();
    }
    //2.当在使用getInstance的时候，它会去取静态内部类的静态属性，
    //这个时候就会导致静态内部类SingletonInstance进行装载，
    //JVM在装载类的时候是线程安全的，装载只执行一次，并实例化INSTANCE
    public static Singleton getInstance(){
        return SingletonInstance.INSTANCE;
    }
}
```

### 优点

1. 这种方式采用了类装载的机制来保证初始化实例时只有一个线程。
2. 静态内部类方式在Singleton 类被装载时并不会立即实例化，而是在需要实例化时，调用getInstance 方法，才会装载SingletonInstance 类，从而完成Singleton 的实例化。
3. 类的静态属性只会在第一次加载类的时候初始化，所以在这里，JVM 帮助我们保证了线程的安全性，在类进行初始化时，别的线程是无法进入的。
4. 优点：避免了线程不安全，利用静态内部类特点实现延迟加载，效率高
5. 结论：推荐使用.

## 4.9 枚举（饿汉式）

```JAVA
enum Singleton{
    
    INSTANCE;//相当于private static final Singleton INSTANCE=new Singleton();
    public sayOK(){
        System.out.println("hello, Everything is ok");
    }
}
```

这借助JDK1.5 中添加的枚举来实现单例模式。不仅能避免多线程同步问题，而且还能防止反序列化重新创建新的对象。

## 4.10 JDK中源码分析

java.run.Runtime就是经典的单例模式

```java
private static Runtime{//饿汉式
    private static Runtime currentRuntime=new Runtime();
    public static Runtime getRuntime(){
        return currentRuntime;
    }
    ...
}
```

## 4.11 单例模式注意事项和细节说明
1. 单例模式保证了系统内存中该类只存在一个对象，节省了系统资源，对于一些需要频繁创建销毁的对象，使用单例模式可以提高系统性能
2. 当想实例化一个单例类的时候，必须要记住使用相应的获取对象的方法，而不是使用new

## 4.12 使用场景

**需要频繁的进行创建和销毁的对象、创建对象时耗时过多或耗费资源过多(即：重量级对象)，但又经常用到的对象、工具类对象、频繁访问数据库或文件的对象(比如数据源、session 工厂等)**

