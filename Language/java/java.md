# 1. 基础
## 1.1. 编程语言的运行机制
1. 编译 源代码->机器码 平台针对性强
2. 编译型语言


    一次性将源代码编译成机器码，可脱离开发环境，效率高，可移植性（跨平台）不强，eg:C/C++/Kotlin
3. 解释性语言


    使用专门的解释器对源程序逐行解释成特定平台的机器码并立即执行，运行时都需要进行解释。效率低，跨平台容易。eg：js/python
4. java的运行机制

    源程序----javac命令编译---->.class文件（与平台无关的字节码）----java命令（虚拟机jvm）解释----->机器码
## 1.2. java开发准备

1. JDK 

    包含java编译器，JRE和类库等

2. JRE
    = JVM + 核心库

3. JDK > JRE > JVM

    运行java程序只需JRE,仅JVM无法运行。而开发需要JDK

4. 安装JDK

    - 配置环境变量

        变量1，JAVA_HOME:安装路径;

        变量2，Path:%JAVA_HOME%/bin;

        让操作系统找到java，javac命令

5. 保存java源文件
    - <div style='color:red'>
          主文件名<br/>
          如果源文件中有public类，该源文件的主文件名必须与public类名相同<br/>
          如果没有，则以其中一个主类命名。<br/>
          如果主类没有，那么可任意命名。<br/>
          public 类可以被项目中其他包下的类访问到。只需要在使用前 import 其对应的 class 文件。将类名与文件名一一对应就可以方便虚拟机在相应的路径（包名）中找到相应的类的信息。
        </div>

    - 扩展名，以.java结尾。

    - <span style='color:red'>一个java源文件只能有一个public类，可以有多个普通类。也可以包含多个主类</span>

        ```java
        public class test{
          public static void main( string[] args ){
            
          }
        }
        ```

6. 编译源（.java）文件

    <span style='color:red'>javac -d [目录] src文件</span>

    eg：javac -d . HelloWorld.java

    尤为注意 <span style='color:red'>字符间的空格。</span>

    目录用于指定将生成的二进制文件放到哪个目录下（绝对路径（c:/xx/xxx/）或相对路径（‘.’表示当前路目录下）），生成了类名.class文件。

    每个java文件可以包含多个类。<span style='color:red'>java源文件中有n个类就会生成n个.class文件</span>

    常见问题：

    - 编码GBK的不可映射字符：javac -encoding UTF-8 HelloWorld.java，指定编码格式

    

7. 运行java程序（解释.class文件)

    <span style='color:red'>java -cp [类的所在目录] 类名</span>

    -cp后面可以跟相对路径

    如果省略-cp，java命令会按照CLASSPATH环境变量的路径来搜索java类，如果没有，默认当前目录下。如果配置了它，他只会在变量所指的路径中查找.class文件；

    eg: CLASSPATH:.;%JAVA_HOME%/lib/dt.jar;%JAVA_HOME%/lib/tools.jar;

    CLASSPATH环境变量是告诉，JVM去哪里找所运行的java类

    常见问题：

    - **找不到或无法加载主类 HelloWorld.class：不要写后缀名.class，只需要写类名就Ok了**

8. 如果希望一个类是可以运行的，该类中必须有一个入口（包含一个主方法）；
<pre>
    public static void main( string[] args ){
        System.out.println("hi,java");
    }
</pre>

9. javap

   是jdk自带的反解析工具。它的作用就是根据class字节码文件，反解析出当前类对应的code区（汇编指令）、本地变量表、异常表和代码行偏移量映射表、常量池等等信息。

## 1.3 基础语法

### 1.3.1 java注释

- 单行注释//
- 多行注释 /*   */
- 文档注释 /** */ &emsp;只能放在类定义之前，五大成员之前。好处：javadoc命令可以直接提取文档注释，并根据文档注释来生成API文档
### 1.3.2 特殊符
- 分隔符：;|{}|[]|()|&nbsp;|.

    语句结束用分号‘;’

- 标识符：字符，数字，下划线，$组成，数字不能开头。

- 关键字和保留字：53个。
### 1.3.3 运算符

- 算术运算符
- 赋值
- 比较
- 逻辑
- 位
  - &、|、~
  - <<、>>、>>>(无符号右移)
- 类型相关运算符

### 1.3.4 流程控制 

1. 顺序结构

    从上到下，一行一行顺序执行

2. 分支结构
- if-else结构

  ```java
  if(cond1){
  	...
  }else if(cond2){
  	...
  }
  ...
  else{
  ...
  }
  ```

  

- switch结构

  ```java
  switch(整型，String，枚举){
      case val1 :
            operation1;
            break;
      ...
      default:
           defaultoption;
  }           
  ```
3. 循环结构
```java
while(boolean){ 
...
}
do{
...
}while(boolean)

for(loopvarinit; condition; step){

}
```

控制循环关键字：break/continue/return

### 1.3.5 数据类型

动态类型和静态类型：在运行期间做数据类型检查和在编译期间做数据类型检查。

强类型和弱类型：强类型指的是程序中表达的任何对象所从属的类型都必须能在编译时刻确定。

<span style='color:red'>java语言是强类型的语言</span>
    

    先声明再使用

<pre>
├── 基本数据类型                  
│   ├── 整数类型 
|   |   ├── byte（1字节）
|   |   ├── short（2）
|   |   ├── int（4）默认
|   |   └── long（8）
|   |  
│   ├── 浮点类型 
|   |   ├── float（4）,eg:3.4f
|   |   └── double（8）默认,eg:3.4
|   |  
│   ├── 布尔类型 
|   |   └── true or false
|   |  
│   └── 字符类型
|       └── char（正整型）
│            
└── 引用数据类型
    ├── 类  
    ├── 接口
    └── 数组                
</pre>

<span style='font-size:20px'>基本类型</span>

整数的存储

所有数值在计算里都是以补码的形式保存的
- 源码：直接换算的二进制码

- 反码：除符号位不变，其他位都取反

- 补码：负数补码=反码+1；正数补码=源码

自动类型转换

    表数范围小——>表数范围大的转换（自动）
    
        byte->short->int->long->float->double
               char---↑

强制类型转换

    表数范围大——>表数范围小的转换（手动）,可能丢失精度。
    格式：typemin var0= (typemin) var1;
<span style="color:red">不用上述强转格式进行转型会报错。</span>

    表达式类型的自动提升，规则：整个表达式的数据类型，与表达式中最高等级的运算数的类型相同
    int it=10;
    it=it/4;不报错，it=2
    double dl=it/4;不报错，it=2.0
直接量
    
    只有三种类型，基本类型，字符串类型和 null类型（只能赋值给引用类型的变量，它是所有引用类型的实例）

### 1.3.6 <span style='font-size:20px'>引用类型--数组</span>

<span style='color:red'>所有已有类型（基本类型+引用类型）加一个[ ]——type[ ]，就变成了新的数组类型</span>


    声明数组变量格式：
    
    type[] arr;
    
    eg:
    int->int[]  新类型，可创建变量，可强制类型转换
    String->String[]
    int[]->int[][]

<span style='font-size:20px;color:red'>java数组的两大特征</span>

1. java语言是强类型的，一个数组只能存储一种数据类型的数据
2. java语言是静态的，java的数组一旦被初始化之后，他的长度就是固定的。

注意：定义数组时，不能指定数组的长度。

数组变量只是一个引用，因此声明时只是定义了一个引用变量，并未真正指向有效的数组对象，也就是并未真正指向有效的内存，因此不能声明长度，而且不能使用

### 1.3.7 数组初始化

必须先声明后初始化，再使用

数组变量只是一个引用，必须让它指向有效内存后才能使用。

1. 静态初始化


        new type[]{ele1,ele2,ele3,...};
        只指定数组的元素，让系统来决定数组的长度。
        eg:int[] iarr=new int[]{1,3,5,7,9}

2. 动态初始化

        new type[length];
        只指定了数组的长度，让系统来决定数组元素的值。
        
        如果type为基本类型，那么系统决定的初始值为 0 / 0.0 / false /\u0000。
        如果数组元素是引用类型，那么所有数组元素的值都是null。
        一旦数组初始化完成，接下来每个数组元素就可当成普通变量使用

3. 声明数组并初始化
         
        type[] arr = new type[length];
        type[] arr = new type[]{ ele1 , ele2 , ele3 ,....} 
        type[] arr= { ele1 , ele2 , ele3,... }//只能在定义数组变量，并初始化时使用
    注：1.数组一旦创建，其长度是固定的（它在内存中位置也是固定的），arr这个引用变量是可以变更的

    ​	2.每个数组元素，就相当于一个变量。可赋值可引用

4. 数组使用

    arr[index]

    数组长度属性：length

    索引范围：0<= index < length

    索引不在范围内：java.lang.ArrayIndexOutOfBoundsException(数组索引越界异常)

    数组没有初始化就使用：java.lang.NullPointerException(空指针异常)

5. 遍历数组（foreach）

    ```java
    for(type var : array | set){
        //对var赋值，不会改变原数组的值
        //如果要再遍历时，对数组元素进行赋值，那就应该根据数组元素的索引来遍历
    }
    ```
### 1.3.8 二维数组

java中只有一维数组，二维数组在实质上是不存在的。

二维数组type\[][]无非是type[]类型一维数组，内嵌int类型一维数组形成。

二维数组的本质是数组为一维数组的数组

```java
public class TDArray{
  public static void main(String[] args){
    //动态初始化，
    //数组元素是int[]（引用类型），所以动态初始化为null
    int[][] arr = new int[5][];
    
    //静态初始化
    int[][] arr1= new int[][]{
                          new int[3],//动态
                          new int[]{1,2,3}//静态
                        };
    int[][] arr2=new int[2][3];

    System.out.println(arr.length)//5
    for(int[] item : arr){
      System.println( item );//将打印出5个null
    }
    
    arr[0]=new int[]{1,2,3,4};
    arr[1]=new int[5];
    for(int i=0 ; i<arr.length ; i++){
      for(int j=0; j< arr[i].length; j++){
        System.out.println(arr[i][j]);
      }
    }
  }
}
```



## 1.4 java内存机制

java的内存可分为：

1. 堆（heap）内存：java虚拟器启动时分配的一块永久的、很大的内存区，堆内存只有一个。new出来的东西，都放在堆内存中。堆内存中的对象，如果没有引用变量指向它，那么它将被JVM的GC(垃圾回收机制回收掉)

2. 栈（stack）内存：每次方法运行时分配一块临时的、很小的内存区。每个方法都有自己对应的栈区，方法结束时，对应栈区就会被回收。（在方法中定义的、局部变量（不管什么类型)，都会放入对应的方法栈。

   ```java
   public class test{
     public static void main(String[] arg){
       //it 放在栈内存
       int it=20;
       //iArr 放在栈内存，new int[4] 放在堆内存(new 出来的数组对象放在堆内存中)
       //引用类型的变量和对象是两个完全不同的东西
       int[] iArr= new int[4];
     }
   }
   ```

   注：<span style='color:red'>引用类型的变量和对象是两个完全不同的东西</span>

   计算机中的每一个内存单元（Byte），在操作系统中都有一个编号。

   变量赋值的区别：

   1. 基本类型的赋值：直接将该值存入变量所在内存
   2. 引用类型的赋值：将该对象所在的第一个内存单元的编号（内存地址）存入变量所在的内存

# 2. 面向对象(上)
1. 三种类型 
    - 类 （class）
    - 接口（interface）
    - 枚举（enum）
2. 四个修饰符
    - private | protected | public
    - final
    - static
    - abstract
3. 五大成员
    - 成员变量（field）
    - 方法（method）
    - 构造器（constructor）
    - 初始化块
    - 内部类
## 2.1 类
最小的程序单位

1. 是一类对象的统称
2. 对象--‘现实’的类
3. 总诀：a.定义类 b.创建对象 c.调用方法

<h4> 类的作用</h4>

1. 声明变量。所有类、都是引用类型，都可用于声明变量。

   ```java
   public class User{
     int a;
     Role b;//用类声明变量 
     //一个外部类可以在一个类里定义变量，但条件是这个外部类必须在CLASSPATH路径下 or 当前文件目录下，并配合适当的修饰符修饰
   }
   ```

   

2. 调用static修饰的方法或static修饰的成员变量

3. 创建对象

4. 派生子类

**对象可用于：调用无static修饰的方法或成员变量**

## 2.2 定义

### 2.2.1 类( class )

<pre>
[修饰符] class className 
{
    1.成员变量（field）;
    2.方法（method）；
    3.构造器（constructor）；
    4.内部类（nested class）；
    5.初始化块；
}
</pre>
1. 修饰符 public、final | abstract
2. 类名 标识符即可，推荐多个单词首字母大写

### 2.2.2 成员变量( field )

用于描述类或对象的状态和属性——名词或形容词

<pre>
[修饰符] 类型 变量名 [=初始值];  
</pre>

1. 修饰符:public | protected | public 、final 、static
2. 类型：基本类型 | 引用类型
3. 变量名：驼峰写法（camelize）

### 2.2.3 方法（method）

用于描述类或对象的行为——动词

<pre>
  [修饰符] 返回值类型 方法名（形参列表）
  {
   //代码块：定义变量、变量复制、流程控制
   //如果声明了返回值类型，必须有return语句
  }
</pre>

1. 修饰符：public | protected | public 、final | abstract 、static
2. 返回值类型：void，任意类型（基本、引用）
3. 方法名：camel
4. 形参列表：形参类型1 形参名1  [，形参类型2  形参名2]

### 2.2.4 构造器（constructor）

new 调用构造器来创建对象。

<b style='color:red;'>如果你没有为类指定构造器，系统会默认为该类提供一个无参数的构造器。</b>

<pre>
  [修饰符] 构造器名(形参列表)
  {
    //代码块
  }
</pre>

1. 修饰符：public | protected | private
2. 构造器名：<span style='color:red'>构造器名必须与类名一致</span>
3. 判断构造器与方法：构造器名与类名是否一致，是否有返回值

## 2.3 this引用

**this可以出现在<span style='color:red'>非static的方法、构造器</span>中。**

作用：

1. 出现在非static的方法中，this代表方了该方法的调用者。谁调用该方法，this就代表谁。
2. 出现在构造器中，this就代表构造器正在初始化的对象。

this很重要的作用是用于区分方法和构造器中与成员变量同名的局部变量。

```java
public class Apple{
  String color;
  double weight;
  int num=0;
  //构造器
  public Apple(String color,double weight){
    this.color=color;
    this.weight=weight;
  }
  public Import(int addnum){
    this.num=this.num+addnum;
  }
}
```

## 2.4 方法

### 2.4.1 方法的所属性

所属性：

1. 有static修饰的方法，属于类本身
2. 无static修饰的方法，属于对象本身。

方法一定要有调用者，才能执行。如果你调用同一个类中的方法，可省略调用者。
此时系统会添加默认的调用者：

1. 为非static方法，添加this作为调用者。
2. 为static方法，添加类作为调用者。

### 2.4.2 形参个数可变的方法

<pre>
  [修饰符]  返回值类型 方法名（参数1，参数2，[参数类型... 形参名]）
{
  //代码块
}
</pre>

1. 写法：类型**...** 参数名，本质上相当于数组type[]，但前者的实参可以分散为多个参数的赋值，而后者只能赋一个数组类型的参数
2. 要求：参数个数可变的参数只能作为参数列表的最后一个参数
3. 优点：参数自动对号入座，多余的参数会自动被封装成为数组

### 2.4.3 递归方法

方法里调用自身方法，递归带来了隐式循环。
避免无限递归：递归一定要向已知的方向递归，并且要出现结束的情况。
通常与if语句、循环语句连用。

### 2.4.4 方法重载（overload）

在同一个类中，方法名相同（包括修饰符，返回类型），形参列表不同——<b style='color:red'>两同一不同</b>

在满足上面的情况下，如果修饰符不同或返回值类型不同，这种情况**不算重载，**系统会报“已在类中定义了什么方法”的错误；

```java
//铜鼓方法重载实现参数默认值
public class A{
   public void doA(int a){
   }
   public void doA(){
       this.doA(0);//这里默认传入0，可以近似认为通过重载实现了默认值的设置
   }
}
```



### 2.4.5 方法的传参机制——值传递

值传递——传入的只是参数的副本，并不是参数的本身

1. 实参为基本类型，方法对形参的操作，对实参无影响。
2. 实参为引用类型，形参和实参指向的是同一个内存地址。因此方法通过形参修改指向的内存地址里的值，会影响实参指向的内存地址里的值。

## 2.5 变量

<pre>
 变 量
  |
  ├── 成员变量（类中定义的，无需显式初始化，系统会自动分配初始值）
  |    |
  │    ├── 类变量 （static修饰）
  |    |
  |    └── 实例变量 （无static修饰，与实例共存亡）
  |
  └── 局部变量（方法中定义的，只在方法中有效。必须显式初始化） 
       |
       ├── 形参
       |
       ├── 方法局部变量（方法中有效）
       |
       └── 代码块局部变量（块中有效）
</pre>
**注意：**

1. 类变量，属于类本身，当系统初始化类时，就会为类变量分配空间，并执行初始化
2. 实例变量，属于对象本身，当系统每次创建对象是，就会为实例变量分配空间，并执行初始化
3. java垃圾语法：允许通过对象访问类变量。但实质上java依然会将对象替换成对象所属的类，然后再访问类变量。建议避免这种用法



## 2.6 构造器

### 2.6.1 作用

1. 用于初始化对象。必须使用new来调用构造器，返回一个初始化完成的对象。
2. 一个类至少存在一个构造器。若没有显示指定构造器，系统将默认提供一个无参构造器。

### 2.6.2 构造器重载

构造器的名和类名相同。

**构造器重载——一个类中可以定义多个构造器，构造器名相同，但形参列表不同**

### 2.6.3 this调用

当多个重载的构造器，在一个构造器中调用其他有相同初始化部分的构造器。

1. this引用和this调用的区别：this引用后面紧跟**“.”**，this调用后面紧跟**“(  )”**，
2. 作用：调用同一个类中重载的构造器。
3. 位置：this调用只能出现在构造器中的第一行。

```java
public class Dog{
  String name;
  String color;
  int age;
  public Dog(String name,String color){
    this.name="D"+name;
    this.color="C"+color;
  }
  public Dog(String name,String color,int age){
    this(name,color);
    this.age=age;
  }
}
```

## 2.7 初始化块

<h5>一、语法</h5>

<pre>[修饰符 static | 无 ]{
//初始化块
}

```java
public class Initial{
    
    String name="h1";
    {
      System.out.println("初始化块");
      name="h2";
      age=3;
      
    }
  	int age = 2;
  public Initial(){
    System.out.println("无参构造器");
  }
  public Initial( String name){
    System.out.println("带Stirng的参构造器，参数为：",name);
  }
}
public class InitialTest{
  Initial in= new Initial();
   System.out.println("name:",in.name);//h2,
  System.out.println("age:",in.age);//2
}
```



<h5>二、实例初始化块</h5>——无static修饰

1. 类被编译之后，实例初始化块的代码，会被还原到每个构造器内的所有代码之前。（可以用javap反编译class文件）

2. 作用：将多个构造器前面部分相同的代码可以提取到实例初始化块中
3. 何时执行：调用构造器创建对象时，程序总会先执行实例初始化块。
4. 定义实例变量时指定初始值，在编译后，也被还原到构造器中所有代码之前，成为一条赋值语句，他和实例初始化代码块的还原顺序，依照他们在源代码中的先后顺序。

<h5>三、类初始化块</h5>——有static修饰

1. 作用：负责对类执行初始化，作用和实例初始化代码块相仿。
2. 何时执行：当程序第一次主动使用该类时（除了用类声明变量时，其余基本都算主动使用）
3. 定义类变量时指定初始值，在编译后，也被还原到类初始化块中所有代码之前，成为一条赋值语句，他和类初始化代码块的还原顺序，依照他们在源代码中的先后顺序。

<h5>四、初始化过程</h5>

1. 初始化任何类之前，一定先从Object开始初始化，依次初始化它所有的祖先类，最后才到他自己。
2. 创建任何对象之前，一定先从Object开始调用构造器，依次执行它所有祖先类的 构造器，最后才执行它自己的构造器。

# 3. 面向对象(下)

<b style='color:red'>面向对象的三大特征——封装、继承、多态</b>

## 3.1 封装（package）

包含两方面意思：合理隐藏内部细节，合理暴露操作

### 3.1.1 访问控制符

| 范围\控制符 | private\|类访问符 | default（不写）\|包访问符 | protected\|子类访问符 | public\|公共访问符 |
| :---------: | :---------------: | :-----------------------: | :-------------------: | :----------------: |
|  同一类中   |         √         |             √             |           √           |         √          |
|  同一包中   |         X         |             √             |           √           |         √          |
| 子       类 |         X         |             X             |           √           |         √          |
| 任       意 |         X         |             X             |           X           |         √          |
| 效       果 |     彻底隐藏      |         部分隐藏          |       部分暴露        |      彻底暴露      |

### 3.1.2 包

**为解决类名重复的问题**

**一、java为类定义包：**

1. 第一步，定义包，在源程序的第一行且非注释行书写如下代码，包名要求：域名倒写+项目文件目录结构

```java
 package  packageName;
```

2. 第二步，编译源程序，系统自动将.class文件放在包名对应的目录结构下（-d . 的作用，没有-d .将会在当前文件夹下生成.class文件)。这之后，.class文件必须在包名对应文件目录结构下，方能有效

```java
javac -d . src.java
```

3. 第三步，运行.class文件，一旦为类指定了包名后，使用时应用完整类名(=包名+类名)，包括创建对象等

```java
java packageName.className
```

**二、导包**

1. import 的作用，为了在使用类时，可省略写包名
2. 位置，在package语句之后，类定义之前，可写多个。
3. 格式：

```java
import packageName.className;//每次导入一个类
import packageName.*;//导入指定包下所有类，*只能替代类名，不能替代包名
 //eg:
import java.util.Random;
```

注意：java程序默认已导入java.lang 包下所有类

4. 静态导入，为了在使用被static修饰的变量或方法时，省略包名和类名，java 1.5引入

```java
import static package.className.staticName;//导入一个静态成员
import static package.className.*;//导入指定类的所有静态成员
//eg:
import static java.lang.System.*;//打印由此可写为out.println('hello world');
import static java.lang.Math.*;//由此写数学方法可写为round(-4.5);
```

**三、java源程序的结构**

1个package 语句，n个import语句，n个class定义（1个public类，多个普通类）

### 3.1.3 封装原则

1. 成员变量，通常用private修饰，
2. 为每个希望隐藏细节、又对外暴露的private修饰的成员变量提供用public修饰的set和get方法，以供给外界使用和控制。
3. 视使用范围的大小合理使用修饰符

## 3.2 继承（extend）

java的继承是类与类之间的关系，由一般到特殊的关系，子类与父类（小类与大类，派生类与基类）

### 3.2.1 格式

<pre>
  [修饰符] 类名 [ extends 父类 ]
  {
  ...
  }
</pre>

注意：

1. **java类只能有一个直接父类（单继承），可以有多个间接父类，如果不显式继承父类，java默认继承Object类**
2. **子类继承父类，可以得到父类的成员变量和方法（当然需要访问控制符至少大于protected）**。
3. <b style='color:red'>访问原则：就近原则。使用时，先在自身类里查找所用变量、方法，然后才往父类里查找。(this引用和super限定皆是如此）</b>

### 3.2.2 方法重写（override）

当子类发现父类的方法不适合自己时，就要重写父类的方法，以适应差异性的需要。

**一、重写条件**

**口诀：2同2小1大**

1. 方法名相同，形参列表相同
2. 返回值类型，声明抛出的异常相同或更小
3. 访问的权限相同or更大

**二、格式**

直接按照重写条件进行重写即可。

如果在重写的方法前 添加@override注解（用来提示报错的），如果该子类没有重写方法，系统会报错。

 ### 3.2.3 super

引用都与实例有关，调用都与构造器有关，this与当前类有关，super与父类有关。

**一、super限定（引用）**

用于限定访问父类定义的实例（实例变量or实例方法）——super**.**field，super**.**method()

```java
//基类
class Base{
  int age=200;
}
//派生类
class Sub extends Base{
  int age=20;
  public void outAge(){
    int age=2;
    System.out.println(age);//2
    System.out.println(this.age);//20
    System.out.println(super.age);//200
  }
}
//程序入口类
class Test{
  public static void main(String[] args){
    Base b = new Base();
    b.outAge();
  }
}
```

**二、super调用**

调用都只能出现在构造器的**第一行，super调用和this调用不能同时出现**

子类一定会调用父类的构造器一次。

通常有以下两种方式：

1. 如果子类构造器没有显式调用父类构造器，系统会自动在子类构造器的第一行调用**父类无参数**的构造器。
2. 子类构造器的第一行显式使用super调用来调用父类构造器

```java
class Fruit{
  double weight;
  public Fruit(double weight){
    this.weight=weight;
  }
}
public class Apple extends Fruit{
  
}//程序在这里会报错，违反了第一个方式，父类没有无参数的构造器
```

 **推论：如果父类没有无参构造器，那么子类必须super显式调用父类指定的构造器。**

## 3.3 多态(polymorphism)

——同一类型的多个实例，在执行同一方法时，呈现出多种行为特征

java引用变量有两个类型：

- 一个是编译时类型，一个是运行时类型。
- 编译时类型由声明该变量时使用的类型决定
- 运行时类型由实际赋给该变量的对象决定。
- 如果编译时类型和运行时类型不一致，就可能出现多态

<h5>一、转型</h5>

基本类型之间的转换只能在数值类型之间进行。

引用类型之间的转换只能在具有继承关系的两个类型之间进行。

1. 向上转型：子类对象可直接赋值给父类变量（自动完成）；

2. 向下转型：父类变量不可直接赋值给子类变量，需要强制转换（<b style='color:red'>格式：(类型)  变量名</b>）、

   出现多态的原因：**Java执行方法时，方法的执行是动态绑定的，方法总是执行该变量实际所指向对象的方法（只针对非静态的方法） 。**

```java
public class PolymorphismTest1{
  public static void main(String[] args){
    //向上转型:子类对象赋值给父类变量，别人要一个鸟，你可以给他一只燕子
    Bird b1 = new Sparrow();//鸟<-燕子
    Bird b2 = new Ostrich();//鸟<-鸵鸟
    b1.fly();//我在天上飞，b1实际指向燕子的实例，燕子实例的fly是继承鸟的fly
    b2.fly();//我在地上跑，b2实际指向鸵鸟的实例，鸵鸟实例的fly用的是自身的重写后的fly
  }
}
public class Bird{
  public void fly(){
    System.out.println("我在天上飞")
  }
}
public class Sparrow extends Bird{
	//直接继承了fly方法
}
public class Ostrich extends Bird{
  //重写了fly方法
  @Override
  public void fly(){
    System.out.println("我在地上跑");
  }
}
```

<h5>二、变量的类型</h5>

1. 编译时类型：声明该变量时，指定的类型。在java程序的编译阶段，Java编译器只认编译时类型
2. 运行时类型：该变量实际所引用对象的类型。

```JAVA
class Shape{
  public void draw(){
    System.out.println("在平面上绘制图形");
  }
}
public class Rect extends Shape{
  public void info(){
    System.out.println("我是一个矩形");
  }
}
public class Circle extends Shape{
  public void round(){
    System.out.println("绕着圆形走一圈");
  }
}
public class PloymorphismTest2{
  public static void main(String[] args){
    Shape s1 = new Rect();//s1编译时类型Shape，
    s1.draw();
    s1.info(); //编译报错，在编译阶段，s1是Shape类型，这一时期，它根本没有info方法，故编译时无法通过。解决办法只有通过：反射，但现在无法解决
    
    Shape s2 = new Circle();
    
    //向下转型，别人要圆圈，你不能给别人一个图形，你要将图形强转为圆圈，然后给别人。
    Circle c1 = (Circle) s2;
    c.round();//不会报错
    
    Circle c2 = s1;//编译报错，编译类型c2为Circle，s1为shape，向下转型失败。
    //替换上面语句
     Circle c2 = （Circle）s1;//运行报错，s1运行时类型是Rect，c2是Circle，
    //如果在编译类型具有继承关系的变量之间进行转换时，如果右侧被转变量的实际类型，与要左侧要转的目标类型不一致，程序就会引发classCastException
    
    //上面更为安全的做法，使用instanceof
    if(s1 instanceof Circle){
      Circle c2 = (Circle) s1;
      c2.round();
    }
    
    //
    String s="fkit";
    System.out.println( s instanceof Integer)//报不兼容的类型，instanceof只能在编译类型具有继承关系之间进行判断
  }
}
```

注意：

1. 向上转型时，父类变量调用的方法必须是父类里有的方法，编译才能通过，否则报错。
2. 编译时，强转运算符只能在编译类型具有父子关系的变量之间进行强转，否则报不兼容类型。
3. 运行时，如果在编译类型具有继承关系的变量之间进行转换时，如果右侧被转变量的实际类型，与要左侧要转的目标类型不一致，程序就会引发classCastException

<h5>三、instanceof</h5>

为了避免classCastException。

变量名 instanceof  类型；

当变量（所引用的对象）为后面的类or子类的实例（我补充：还有后面类的父类的实例，不知对不对，后面要验证）时，返回true；

instanceof只能在编译类型具有继承关系之间进行判断，否则编译报错：不兼容的类型

## 3.4 修饰符

### 3.4.1 static

**一、使用**

1. 可static修饰的成员：成员变量、方法、初始化模块和内部类，而不可用static修饰构造器、局部变量（根本不属于类的五大成员）、外部类。

2. **static成员不可直接访问非static成员（借用类或对象去调用），而非static成员可访问static成员（祖先的物品和自己的物品）**

   ```java
   public class StaticTest{
     int age=20;
     public info(){
       System.out.println("这是info方法")；
     }
     class A{
       
     }
     public static void mian(String[] args){
       System.out.println(age);//无法从静态上下文中引用非静态变量age
       info();//无法从静态上下文中引用非静态变量info();
       A a = new A();//无法从静态上下文中引用非静态变量this
     }
     int b = 10；
     static int c= b ；//无法从静态上下文中引用非静态变量b
   }
   ```

   

### 3.4.2 final

可修饰各种变量，方法，类。

final与abstract是互斥的。

<h5>1. 修饰变量</h5>

 **final修饰变量：变量必须赋初始值，该变量被赋初始值之后，不能被重新赋值。**


1. 修饰成员变量：必须显示指定初始值，只能在以下几个位置的其中之一指定

   - 实例变量：非静态初始化块，声明处，构造器三个位置之一。这三个位置的本质其实只有一个：构造器
   - 类变量：静态初始化块，声明处，这两个位置本质其实只有一个：静态初始化块

2. 修饰局部变量：和非final修饰的变量相同，都必须先指定初始值，然后才能使用，只是final修饰的局部变量不可被重新赋值

3. 修饰引用类型变量：只保证这个引用类型的变量所引用的地址不变，可改变其引用的对象

4. 修饰“宏替换”变量：

   - 宏替换变量定义：如果一个方变量满足

     - 有final修饰

     - 声明时指定初始值

     - 初始值可以在编译时确定下来（可以使用算数表达式和字符串连接符，没有访问变量或方法，那么这个初始值就可以在编译时确定下来，放入常量池）

       那么这个变量就会消失，所有出现该变量的地方，在编译时，都会替换成该变量的值。

       第一次使用某一个直接量，他会进入常量池，当再次使用时，会直接指向前面的常量池。

     ​	

<h5>2. 修饰方法

final修饰的方法不可被重写，可重载，可被子类调用

<h5>3. 修饰类


final修饰的类，不能派生子类

### 3.4.3 abstract

接口和抽象类的价值在于设计和规范

只能修饰方法和类，final与abstract是互斥的。

1. 抽象类——主要用于派生子类，调用类方法和类变量，定义变量（只能用子类实例赋值）
   - 抽象类与普通类的区别：
     - 得到一个新功能：拥有抽象方法，只有抽象类才能有抽象方法
     - 失去一个功能：不能创建对象，但存在构造器
2. 抽象方法——只有方法签名，没有方法体，不能有**{}**
3. 子类规则：要么重写抽象父类中所有抽象方法，要么子类也是抽象的。

## 3.5 接口

接口相当于一种更为抽象的类

体现的是一种规范——暴露出来供大家遵守，接口里所有东西都默认有public修饰

### 3.5.1 定义

```java
[修饰符 public ] interface 接口名 extends 父接口1，父接口2,...{
    
    //三大成员：
    //成员变量。只有常量，始终会添加public static final修饰，通常不写
    //抽象方法。java8 之后，可以写类方法，默认放飞
    //内部类
    //没有初始化块，没有构造器
}
```

### 3.5.2 接口的用处

1. 定义变量（只能用实现类的实例赋值向上转型）
2. 调用类方法or类变量
3. 派生实现类

### 3.5.3 实现接口

```java
[public] class 类名 extends 父类 implements 父接口1,父接口2..{
    
    //五大成员
    
}
```

实现类要么重写接口中所有的抽象方法，要么也只能是抽象类

重写接口中的方法，只能用public修饰

接口不可继承接口，类也不能继承接口，同样，接口不能继承类

## 3.6 枚举

在某些情况下，一个类的对象是有限且固定的。

它与class、interface的地位相同。使用enum定义的枚举类默认继承了java.lang.Enum，间接继承Object类。**枚举类不能显式继承其他父类。**

枚举类是一种特殊的类，他一样可以拥有自己的成员变量、方法，可以实现一个或则多个接口，也可以定义自己的构造器。

```java
public enum Gender{
    //1.所有枚举实例只能在第一行显式列出，否则不能产生实例，默认添加修饰符public static final，不能使用其他修饰符。
    MALE("男"),FEMALE("女");//这里相当于private static final Gender MALE=new Gender("男");
    
    //如果枚举类里定义了抽象方法，那么必须显示为每个枚举值提供抽象方法实现
    // MALE("男"){
    //     public String label(String adj){
    //         return adj+this.name
    //     }
    // },
    // FEMALE("女"){
    //     public String label(String adj){
    //         return adj+this.name
    //     }
    // }
    
    //枚举类同样可以实现接口
    
    //2.可以定义成员变量
    private final String name;
    //3.构造器默认private修饰，显式只能使用private。
    private Gender(String name){
        this.name=name;
    }
    //4.方法
    public String getName(){
        return this.name;
    }
    //5.抽象方法
    public abstract double label(String adj);//枚举值要提供实现
    
    //6.枚举类的使用
    public static void main(String[] args){
        System.out.println(Gender.MALE.getName());
        System.out.println(Gender.FEMALE.getName());
    }
}


//枚举类的使用

```



# 4. 注解(Annotation)

注解是代码里的特殊标记，这些标记可以在编译、类加载、运行时被读取，并执行相应的处理。

注解不影响程序代码的执行，如果希望让程序中的注解在运行时起一定的作用，必须使用注解的工具APT(Annotation Processing Tool)来处理

通过使用注解，开发人员可以在不改变原有逻辑的情况下，在源文件中嵌入一些补充信息（ **@Annotation(name1=value1,...) ，以name-value对的形式存储补充信息**）。代码分析工具、开发工具和部署工具可以通过这些补充信息进行验证或者进行部署

## 4.1 基本注解

- @Override-强制一个子类必须覆盖父类的方法
- @Deprecated-用于表示某个程序元素（类，方法等）已过时
- @FunctionalInterface-指定某个接口必须是函数式接口
- @SupressWarnings-指定元素取消显示指定的编译器警告

## 4.2 元注解

java.lang.annotation包下提供了6个meta注解（元注解），其中有5个注解都用于修饰其他的注解定义

- @Retention-用于指定被修饰的注解可以保留多长时间
- @Target-用于指定被修饰的注解能用于修饰哪些程序单元
- @Documented-用于指定被该注解修饰的注解类将被javadoc工具提取成文档
- @Inherited-用于指定被它修饰的注解类具有继承性

## 4.3 自定义注解

### 4.3.1 定义注解

```java
public @interface MyTag{
    //还可以带成员变量
    //成员变量在注解定义中以无形参的方法形式来声明
    //一旦注解里定义了成员变量，使用该注解时就应该为它的成员变量指定值
    String name() default "qin";//还可以指定默认值
    int age();
}
public class Test{
    @MyTag(name="kang",age=24)
    public void info(){
        
    }
}
```

### 4.3.2 提取注解信息

java使用java.lang.annotation.Annotation接口来代表程序元素前面的注解，该接口是所有注解的父接口。

java 5 在java.lang.reflect包下新增了AnnotatedElement接口。它是所有程序元素的父接口，所以程序可以通过反射获取了某个类的AnnotatedElement对象之后，程序就可以调用该对象的如下几个方法来访问注解信息。

详见疯狂java讲义p662

# 5. 集合

java集合大致可分为Set、List、Queue和Map四种体系。

- Set代表无序、不可重复的集合；常用：HashSet、TreeSet
- List代表有序可重复的集合；常用：ArrayList、LinkedList
- Map则代表具有映射关系的集合；常用：HashMap、TreeMap
- ava5又增加了Queue体系集合代表一种队列集合实现；常用：ArrayDeque

![](./legend/java三种集合的示意图.png)

java集合就像一种容器，可以把多个对象（实际上是对象的引用，但习惯上都称对象）“丢进”该容器。

数组只能将统一类型的对象放在一起，而集合可以放多种类型的对象。数组可以存放基本类型的值，也可以存放对象，而集合只能存放对象。

数组长度不可变，无法存放具有映射关系的数据，

![IteratorTree.png](/legend/IteratorTree.png)

java集合类主要由两个接口派生而出：Collection和Map。

## 5.1 Collection与遍历

### 5.1.1 Collection的公共方法

Collection接口是List、Set、Queue接口的父接口，该接口里面定义的方法都可作用于三种集合

- **boolean add(Object o)**，向集合中添加元素
- **boolean addAll(Collection c)**，将集合c中的所有元素添加到指定集合中
- **void clear()**，清除集合中所有的元素
- boolean remove(Object o)，删除集合中指定的元素o
- boolean remove(Collection c)，从集合中删除集合c中包含的所有元素
- boolean retainAll(Object o)，从集合中删除集合c中不包含的元素
- boolean contains(Object o)，返回集合中是否包含指定元素
- boolean containsAll(Collection c ),返回集合中是否包含c集合中的所有元素
- boolean isEmpty()，返回集合是否为空
- int size()，返回集合元素的个数
- Object[] toArray()，将集合转换为一个数组
- Iterator iterator()，返回一个Iterator对象，用于遍历集合里的元素

所有的Collection实现类都重写了toString()方法，该方法可以一次性输出集合中所有元素。

### 5.1.2 集合遍历

#### foreach

```java
public class ForeachTest{
    public static void main(String[] args){
        ...
            for(Object o : c){
                //o 只是集合中元素的副本
                if(o.equals("haha")){
                    c.remove(o);//会引发ConcurrentModificationException
                    //集合遍历时，不能对原集合进行修改,
                }
            }
    }
}
```

#### 使用Lambda表达式遍历集合

```java
puclic class LambdaEach{
    public static void main(String[] args){
        Collection books = new HashSet();
        books.add("道德经");
        books.add("自卑与超越");
        books.add("理想国");
        books.forEach( obj -> System.out.println("current object:"+obj));
        
    }
}
```

java8 为Iterator接口新增了一个forEach(Consumer action)默认方法，该方法所需参数的类型是一个**函数式接口**，forEach方法会自动将集合元素逐个地传给Lambda表达式的形参。

#### 依靠Iterator遍历集合

```java
public class IteratorEach{
    public static void main(String args){
        //...
        Iterator it = books.iterator();
        while(it.hasNext()){
            String book = (String) it.next();
            if(book.equals("理想国")){
                it.remove();//可以通过Iterator的remove()方法删除上一次next返回的集合元素
        }
    }
}
```



## 5.2 Set

Set集合通常不能记住元素的添加顺序，不允许包含相同的元素。有HashSet，TreeSet和EnumSet三个实现类。

### 5.2.1 HashSet

HashSet不是同步的，线程不安全。

当向HashSet集合中存入一个元素对象时，HashSet会调用该对象的hashCode方法来得到该对象的hashCode值，然后根据该hashCode值决定该对象在HashSet的存储位置。

HashSet集合判断两个对象相等的标准是两个对象通过equals()方法比较相等，并且两个对象的hashCode()方法返回也相等。如果对象相等那么那么不能重复添加。

所以重写equals和hashCode方法时，应做到equals返回为true时，hashCode函数返回的HashCode值也应相同。反之亦然。

HashSet中每个能存储元素的“槽位（slot）”通常称为“桶（bucket）”，如果有多个元素hashCode值相同，而equals返回的值却为false，就需要在一个桶里放多个元素（因为hashCode值决定存储位置），这样会导致性能下降。

#### LinkedHashSet

HashSet的子类，LinkedHashSet同样根据hashCode值来决定元素的存储位置，但它同时使用链表维护元素的次序，当遍历LinkedHashSet集合中的元素时，将会按照元素的添加顺序来访问集合中的元素。

插入时，LinkedHashSet由于要维护元素的插入顺序，所以插入性能略低于HashSet。

遍历时，LinkedHashSet可以通过Link来访问，所以访问性能要高于HashSet。

### 5.2.2 TreeSet

TreeSet是SortedSet接口的实现类，TreeSet可以确保元素处在排序状态。TreeSet还提供了额外的几个方法

- Object first()/last()：返回集合的第一个（最后一个）元素
- Object lower( Object e )/higher( Object e )：返回集合中小于/大于指定元素之前/之后的元素（即小于/大于指定元素的最大/最小元素），指定元素可以不是TreeSet集合中的元素
- SortedSet subSet( Object formElement, Object toElement )
- SortedSet headSet( Object toElement )
- SortedSet tailSet( Object formElement )

TreeSet支持两种排序方法：自然排序和定制排序

#### 自然排序

TreeSet会调用集合元素的compareTo( Object obj ) 方法来比较元素之间的大小关系，然后将集合元素按升序排列。

如果试图将一个对象添加到TreeSet时，需注意：

- 该对象必须实现Comparable接口，否则会报classCastException异常。
  - 该接口定义了一个`compareTo(Object obj)`方法，该方法返回一个整数值。例如`obj1.compareTo(obj2)`，如果该方法返回0，则两对象相等，返回正整数，表明obj1大于obj2，返回负整数，表明obj1小于obj2。【0等正大负小】
  - 元素对象相等的唯一标准是`obj1.compareTo(obj2) = 0`
  - equals()返回True时，compareTo()应返回0
- 所有元素对象应该是同一个类的对象

TreeSet可以删除没有被修改实例变量、且不与其他修改实例变量的对象重复的对象。

#### 定制排序

如果要实现定制排序，则需要在创建TreeSet集合对象时，提供一个Comparator对象与该TreeSet集合关联，由该Comparator对象负责集合元素的排序逻辑。

由于Comparator是一个函数式接口，因此可以使用lambda表达式代替Comparator对象。

```java
//...
TreeSet ts = new TreeSet((o1, o2) -> {
    M m1 = (M) o1;
    M m2 = (M) o2;
    return m1.age > m2.age ? -1 :(m1.age < m2.age ? 1 : 0)
})
```

### 5.2.3 EnumSet

专门为枚举类设计的集合类，EnumSet以枚举值在Enum类内的定义顺序来决定集合元素的顺序。

EnumSet只能保存同一个枚举类的枚举值作为集合元素。

EnumSet类没有暴露任何构造器来创建该类的实例，程序应该通过它提供的类方法来创建EnumSet对象。

### 5.2.4 性能分析

TreeSet需要额外的红黑树算法来维护集合元素的次序。

HashSet的性能总是比TreeSet好，特别是添加和查询等操作，所以只有当需要保持排序的Set时，才应该使用TreeSet，否则都应该使用HashSet。

LinkedHashSet遍历的性能比HashSet要好，HashSet添加和删除的操作性能比LinkedHashSet要好。

Set的三个实现类都是线程不安全的。

## 5.3 List

List集合代表一个元素有序（元素都有顺序索引，默认按照添加顺序设置元素索引）、可重复的集合。

List集合中增加了一些根据索引来操作集合元素的方法。

- `void add(int index,Object o)`
- `boolean addAll(int index,Collection c)`
- `Objcet get(int index)`
- `int indexOf(Object o)`
- `int lastIndexOf(Object o)`
- `Object remove(int index)`
- `Object set(int index,Object element)`
- `List subList(int fromIndex,int toIndex)`
- `void sort( Comparator c )`
- `void replaceAll()`

List判断两个对象相等，只要通过`equals()`方法比较返回true即可。这主要和List的`remove（Object obj）`有关系。

ArrayList 和 Vector作为List类的两个典型实现，都是基于数组实现的List类。

ArrayList是线程不安全的，Vector是线程安全的。

注意：

- 如果需要遍历List，对于ArrayList应该使用随机访问访问方法俩遍历集合，对于LinkedList（是Deque的实现类）则应该使用迭代器来遍历集合元素
- 经常执行插入删除操作，采用LinkedList性能更佳

## 5.4 Queue

Queue用于模拟队列这种数据结构，队列通常是指“先进先出FIFO"的容器。

Queue接口定义了如下几个方法：

- 向队尾添加元素（入队列）：

  - void add( Object o)

  - boolean offer(Object o)：将指定元素加入到队列的尾部。当使用有容量限制的队列时，此方法通常比add方法更好

- 获取队首的元素：

  - Object element()：获取队列头部的元素，但不是删除元素
  - Object peek()：获取队列头部的元素，但不是删除元素，如果队列为空则返回null

- 获取队首的元素并出队列

  - Object poll()：获取队列头部的元素，并删除元素，如果队列为空则返回null
  - Object remove()：获取队列头部的元素，并删除元素

Queue接口有一个PriorityQueue实现类。

除此之外，Queue还有一个Deque接口。

Deque代表一个双端队列，可以同时从两端来添加、删除元素，因此Deque的实现类既可当成队列来使用，也可当成栈来使用。

Jave为Deque提供了ArrayDeque和LinkedList两个实现类。

### 5.4.1 PriorityQueue 

PriorityQueue优先权队列，保存元素的队列不是按照入队的顺序，而是按照队列元素的大小进行排序，出队是取队列中最小的元素。和TreeSet一样存在两种排序方式，具体可参照TreeSet的排序。

### 5.4.2 Deque

Deque接口是Queue接口的子接口，代表双端队列。Deque定义了一些双端队列的方法。

- 双端添加：`void addFirst(Obj)、void addLast(Obj)、boolean offerFirst(Obj)、boolean offerLast(Obj)`
- 双端移除：`Object removeFirst(Obj)、Object removeLast(Obj)、Object pollFirst(Obj)、Object pollLast(Obj)`、如果队列为空poll返回null
- 双端获取：`Object getFirst()、Object getLast()、Object peekFirst()、Object peekLast()`，如果队列为空，peek返回null
- 栈：`Object pop()、void push(Obj)`
- 迭代器方法：`Iterator descendingIterator()`，返回迭代器，该迭代器将以逆向顺序来迭代队列中的元素
- 出现：`removeFirstOccurrence(Obj)、removeLastOccurrence(Obj)`，删除第一次、最后一些出翔的元素obj

队列方法：

- 入队列（插入到队尾）：`addLast、offerLast`
- 出队列（从队首出）：`removeFirst、pollFirst`，如果队列为空，pollFirst返回null
- 获取队首元素（获取队首元素但不删除）：`getFirst、peekFirst`，如果队列为空，peekFirst 返回null

栈方法：

- 入栈：`push、addFirst、offerFirst`
- 出栈：`pop、removeFirst、pollFirst`
- 获取栈顶元素：``getFirst、peekFirst`，如果队列为空，peekFirst 返回null`

Deque的实现类：

- ArrayDeque，基于数组实现的双端队列，有较好的随机访问性能
- LinkedList，即是List的实现类（因此可以基于索引来随机访问），又是基于链表实现的双端队列，有较好的插入、删除性能

## 5.5 Map

Map用于保存具有映射关系的数据（key-value对），key和value都可以是任何引用类型的数据。

一个Map对象的任何两个key通过equals方法比较总是返回false。

如果把map里的所有Key放在一起来看，他们就组成了一个Set集合。

如果map里的所有value放在一起来看，又非常类似于一个List，每个元素可根据索引来查找，Map中的索引为另一个对象（而不再是整数值）

Map接口下则有HashMap、LinkedHashMap、SortedMap、TreeMap、EnumMap等等和Set有相似的子接口和实现类，正如名字所暗示的，Map的这些实现类和子接口中的key集的存储形式和对应Set集合中元素的存储形式完全相同。

- `void clear()`
- `boolean containsKey(Object key)、boolean containsValue(Object value)`
- 

# 6. 泛型

一旦把一个对象丢进java集合中，集合就会忘记对象的类型，把所有对象当成Object类型处理，当程序从集合中取出对象后，就需要进行强制类型转换，这种转换不仅使代码臃肿，而且容易引起classCastException异常。

## 6.1 使用泛型

```java
List<String> strList=new ArrayList<String>()
    //在集合接口或集合接口的实现类后增加尖括号，尖括号里放一个数据类型，即表明这个集合接口、集合类只能保存特定类型的对象。
Map<String,Integer> scores= new HashMap<String , Integer>();
```

### 6.1.2 java7 增强的"菱形语法"

```java
//菱形类推断
Map<String,Integer> scores= new HashMap<>();
//java7开始，允许在构造器后不需要带完整的泛型信息，只要给出一对尖括号，它可以推断括号里应该是什么泛型信息
```

## 6.2 深入泛型

所谓泛型，就是允许在定义类、接口、方法时使用类型传参，这个类型形参（或叫泛型）将在声明变量、创建对象、调用方法时动态地指定（即传入实际的类型参数，也可称为类型实参）、泛型形参在整个接口、类体可当成类型使用。

```java
//定义接口时，指定了一个泛型形参，该形参名为E
public interface List<E>{
    //在该接口中，E可以作为类型使用
    void add(E x);
}
//定义接口时，指定了两个泛型形参，其形参名为K,V
public interface Map<K,V>
{
	Set<K> keySet()
	V put(K key, V value)
}
public class Apple<T>
{
	private T info;
	public Apple(){}
	public Apple(T info){ this.info = info;}
	public void setInfo(T info){ this.info = info;}
	public T getInfo(){ return this.info;}
    public static void main(String[] args){
    	Apple<String> a1 = new Apple<>("苹果");
    	System.out.println(a1.getInfo());
    	Apple<Double> a2 = new Apple<>(12.12)
    	System.Out.println(a2.getInfo());
    }
}
```

例如在使用List类型时，如果为E形参传入String类型实参，则产生了一个新的类型：List\<String>

泛型类也可以派生子类：

- 定义泛型类、接口、方法时可以声明泛型形参
- 而使用泛型类、接口、方法时应该为泛型形参传入实际的类型

```java
//定义时
public class Apple<T>{}
//使用泛型类、接口时可以不传，但建议传
public class redApple extends Apple<String>{}
public class greenApple extends Apple{}//省略泛型的形式会被默认为raw type，编译器可能会发出警告
```

系统中并不会真正生成泛型类，不管泛型类的实际类型参数是什么，它们在运行时总是有同样的类。

```java
List<string> l1 = new ArrayList<>();
List<Integer> l2 = new ArrayList<>();
System.out.println(l1.getClass() === l2.getClass())//true
```

在静态初始化块或静态变量的声明和初始化块不允许使用泛型形参。

## 6.3 类型通配符

假设Foo是Bar的一个子类，那么`Foo[]依然是Bar[]`的子类，带`G<Foo>不是G<Bar>的子类`

为了表示各种泛型的父类，可以使用类型通配符。

类型通配符是一个问号`?`，它的元素类型可以匹配任何类型。

### ? extends superClass

```java
List<?> c = new ArrayList<String>();
c.add(new Object())	//引发编译报错
    
//通配符的上限
public abstract class Shape{
    public abstact void draw(Canvas c);
}
public class Circle extends Shape{
    public void draw(Canvas c){
        System.out.println("在画布"+ c + "画个圆");
    }
}
public class Rectangle extends Shape{
    public void draw(Canvas c){
        System.out.println("在画布"+ c + "画个长方形");
    }
}
public class Canvas{
    //错误方法
    public void drawAll(List<Shape> shapes){
        for(Shapes s : shapes){
            s.draw(this);
        }
    }
    //正例方法，
    public void drawAll(List<? extends Shape> shapes){
        //通过使用通配符的上限，即List<? extends Shape>可以被表示List<Circle>、List<Ractangle>的父类
        //这种指定通配符上限的集合，只能从集合中取元素（取出的元素总是上限的类型）
        for(Shapes s : shapes){
            s.draw(this)
        }
    }
    public void addRectangle(List<? extends Shape> shapes){
        //下面将引起报错
        shapes.add(0,new Rectangle());
        //shapes.add()的第二个参数类型是? extends Shape，它表示Shape未知的子类，程序无法确定这个类型是什么，所以无法将任何对象添加到集合中。
        //(因为编译器没法确定集合元素实际是哪种子类型)
        
    }
}
public class Test{
    public static void main(String[] args){
        List<Circle> circleList = new ArrayList<>();
        Canvas c = new Canvas();
        //如果用错误的方法，则不能将List<Circle>当成List<Shape>使用，所以下面代码引起编译错误
        c.drawAll(circleList)
    }
}
```

指定通配符上限就是为了支持类型型变，Foo是Bar的子类，这样`A<Foo>就相当于A<? extends Bar>的子类`，可以将`A<Foo>赋值给A<? extends Bar>类型的变量`，这种型变方式称为协变。

对于协变的泛型类来说，它只能调用泛型类型作为返回值类型的方法（编译器会将该方法返回值当成通配符上限的类型），而不能调用泛型类型作为参数的方法，口诀是：协变只出不进。

### ? super subClass

指定通配符的下限就是为了支持类型型变，比如Foo是Bar的子类，当程序需要一个`A<? super Foo>变量时，程序可以将A<Bar>、A<Object>赋值给A<? super Foo>类型的变量`，这种型变方式称为逆变

对于逆变的泛型集合来说，编译器只知道集合元素是下限的父类型。但具体是哪种父类型则不确定。因此这种逆变的泛型集合能向其中添加元素（因为实际赋值的集合元素总是逆变声明的父类），从集合中取元素时只能被当做Object类型处理（编译器无法确定取出的到底是哪个父类的对象）。

```java
public class MyUtils{
    public static <T> T copy(Collection<? super T> dest, Collection<T> src){
        T last = null;
        for(T ele : src){
            last = ele;
            dest.add(ele);
        }
        return last;
    }
    public static void main(String[] args){
        List<Number> ln = new ArrayList<>();
        List<Integer> li = new ArrayList<>();
        li.add(5);
        Integer last = copy(ln, li);
        System.out.println(ln);
    }
}
```

## 6.4 泛型方法

在定义类、接口的时候没有使用泛型形参，但定义方法时想自定义泛型形参——泛型方法

所谓泛型方法，就是在声明方法时定义一个或多个泛型形参，泛型方法的语法格式如下：

```java
//定义泛型方法的语法格式
修饰符 <T, S> 返回值类型 方法名( 形参列表 ){
	//方法体
    //泛型方法和普通方法格式比较起来，多个泛型形参声明，泛型形参声明以尖括号括起来，多个泛型形参之间以逗号隔开。
    //所有泛型形参声明放在方法修饰符和返回值类型之间
}
public class GenericMethodTest{
    static <T> void fromArrayToCollection(T[] a, Collection<T> c){
        for(T o : a){
            c.add(o);
        }
    }
    public static void main(String[] args){
        String[] sa = new String[100];
        Collection<string> cs = new ArrayList<>();
        //与类和接口中使用泛型参数不同，方法中的泛型参数无须显式传入实际类型参数，编译器可以根据实参进行推断
        fromArrayToCollection(sa,cs);
    }
}

//泛型方法大部分时候可以代替类型通配符
public interface Collection<E>{
    boolean containsAll(Collection<?> c);
    boolean addAll(Collection<? extends E> c);
}
public interface Collection<E>{
    <T> boolean containAll(Collection<T> c);
    <T extends E> boolean addAll(Collection<T> c);
}
```

泛型方法使用场景：

泛型形参被用来表示方法的一个或多个参数之间的类型依赖关系，或者方法的返回值与参数之间的类型依赖关系。如果没有这样类型的依赖关系，就不应该使用泛型方法。

# 7. 类加载与反射

## 7.1 类加载

当程序主动使用某个类，如果该类还未被加载到内存中，则系统会通过：类的加载、类的连接，类的初始化三个步骤来对该类进行初始化，如果没有意外，JVM会连续完成这三个动作，所以有时也把这三个步骤统称为类加载或类的初始化。

### 7.1.1 JVM

当调用java命令运行某个java程序时，该命令将会启动一个JVM（java虚拟机）进程，不管java程序多么复杂，该程序启动了多少个线程，它们都处于该Java虚拟机进程中，他们都是用该JVM的内存区。

系统出现以下情况，JVM进程将被终止：

- 程序运行到最后正常结束
- 程序执行到`System.exit()或Runtime.getRuntime().exit()`代码处
- 程序遇到未捕获的异常或错误
- 程序所在平台强制结束JVM进程

### 7.1.2 类的加载

类的加载指的是将类的class文件读入内存，并为之创建一个`java.lang.Class`对象

类的加载由类的加载器完成，通常由JVM提供（系统类加载器），这些加载器是所有程序运行的基础。除此以外，开发者可以通过继承ClassLoader来创建自己的类加载器

通过使用不同的类加载器，可以从不同来源加载类的二进制数据，通常有以下来源

- 本地文件系统加载class文件
- jar包中加载class文件
- 通过网络加载class文件
- java源文件动态编译，并执行加载。

类加载器通常无需等到“首次使用”该类时才加载该类，JVM规范允许系统预先加载某些类。

### 7.1.3 类的连接

类的连接负责把类的二进制数据合并到JRE中。类连接存在下面三个阶段

- 验证，用于检验被加载的类是否有正确的内部结构，并和其它类协调一致
- 准备，负责为类的类变量分配内存，并设置默认初始值
- 解析，将类的二进制数据中的符号引用替换成直接引用

### 7.1.4 类的初始化

JVM初始化一个类包含如下几个步骤：

- 假如一个类还没有被加载和连接，则程序先加载并连接该类
- 假如该类的直接父类还未被初始化，则先初始化其直接父类
- 假如类中有初始化语句，则系统依次执行这些初始化语句

初始化时机：

- 创建类的实例，创建实例的方式有：new操作符，反射，反序列化
- 调用某个类的静态方法
- 调用某个类的静态变量，或为该静态变量赋值
- 使用反射方式来强制创建某个类或接口对应的java.lang.Class对象
- 初始化某个类的子类，该子类的所有父类都会被执行初始化
- java.exe命令来运行某个主类

如果final的类变量可以在编译时确定值，当程序其他地方使用该类变量时，实际上并没有使用该变量，而相当于使用常量，则不会导致类的初始化。

## 7.2 类加载器

类加载器负责实现类的加载，负责将.class文件（在网络上，或在磁盘上）加载到内存中，并为之生成对应的j`ava.lang.Class`对象

### 7.2.1 类加载机制

一旦一个类被载入到JVM中，同一个类就不会被再次载入了，在java中，一个类用其全限定类名（包括包名和类名）作为唯一标识。但在JVM中一个类用全限定名和其类加载器作为唯一标识。

JVM的类加载机制

- 全盘负责：当一个加载器加载某个Class时，那么该Class所依赖和引用的其它Class都由其负责，除非显示使用另外一个加载器来载入。
- 父类委托：先让父类加载器（加载器的父类）试图加载该Class，只有父类加载器无法加载该类时才会尝试从自己的类路径中加载该类
- 缓存机制：该机制保证所有加载过Class都会被缓存，当程序需要使用某个Class时，类加载器会首先在缓存区中搜寻该Class，如果缓存区不存在，系统才会读取该类对应的二进制数据，并将其转换成Class对象，存入缓存区。

类加载器之间的继承关系不是类继承上的父子关系，而是类加载器实例之间的关系。

![](./legend/类加载器实例之间的关系.png)

```java

public class Test{
    public static void main(String[] args){
        //获取系统类加载器
		ClassLoader systemLoader = classLoader.getSystemClassLoader();
        //获取系统类加载器的父类加载器，扩展类加载器
        ClassLoader extensionLoader = systemLoader.getParent();
    }
```



### 7.2.2 自定义类加载器

JVM中除了根类加载器之外的所有类加载器都是ClassLoader子类的实例，开发者可以通过继承ClassLoader来实现自定义。

java为ClassLoader提供了一个URLClassLoader实现类，它既可以从本地文件系统获取二进制文件来加载类，也可以从远程主机获取二进制文件来加载类。

## 7.3 反射

引用对象存在在运行时存在两种类型：编译时类型和运行时类型，`例如：Person p = new Student();`

该变量编译类型为Person，运行时类型为Student。

当Person类和Student类都存在同样的方法时，在运行时就可以调用到运行时类型的方法（多态）。

当一个方法存在于Student类，而不存在于Person类，那么程序就会报错。

程序在很多时候需要调用运行时类型的方法，并且此时也不知道对象和类属于哪些类，程序只依靠运行时信息来发现该对象的真实信息，这就必须使用反射。

### 7.3.1 获取Class对象

每个类被加载之后，系统就会Wie该类生成一个Class对象，通过该Class对象就可以访问到JVM中的这个类。

在java程序中获得Class对象通常有如下三种方式：

1. 通过类来获取
   - 通过Class类的`forName(String clazzName)`静态方法，clazzName为类的全限定类名
   - 通过某个类的`class属性`，例如：`Person.class`
2. 通过对象来获取
   - `调用某个对象的getClass()`方法

### 7.3.2 由Class对象获取信息

Class类提供了大量的**实例方法**来获取该Class对象所对应类的详细信息。

1. 获取构造器
   - `Constructor<T> getConstructor(Class<?> ... parameterTypes),`返回此Class对象对应类的，带指定形参列表的public构造器
   - `Constructor<?>[] getConstructors(),`返回此Class对象对应类的所有Public构造器
   - `Constructor<T> getDeclaredConstructor(Class<?> ... parameterTypes),`返回此Class对象对应类的所有public构造器、带指定形参。与构造器访问权限无关
   - `Constructor<?>[] getDeclaredConstructors(),`返回此对象所有构造器，与构造器访问权限无关
2. 获取方法，与构造器一致，只需把Constructor换成Method即可
3. 获取成员变量，与构造器一致，只需把Constructor换成Field即可
4. 获取Class对应类上所包含的Annotation、内部类、修饰符、所在包、类名等基本信息
   - `int gitModifiers()`，获取修饰符
   - `Package getPackage()`，获取此类的包
   - `String getName()`，获取类名，`getSimpleName`获取类名的简称
5. Class对象还可以调用几个判断方法来判断该类是否为接口、枚举、注解类型等
   - `boolean isAnnotation(), isAnnotationPresent(Class<? extends Annotation> annotationClass)`
   - `boolean isInterface(), isInstance(Object obj)`
6. 

### 7.3.3 创建并操作对象

Class对象可以获得该类里的方法（由Method对象表示）、构造器（由Constructor对象表示）、成员变量（由Field对象表示），程序可以通过Method对象来执行对应的方法、通过Construcor对象来调用对应的构造器来创建对象，通过Field对象直接访问并修改对象的成员变量值。

这三个类都位于java.lang.reflect包下，并实现了java.lang.reflect.Member接口。

#### 创建

通过反射创建对象步骤：

- 先使用Class对象获取指定的Constructor对象，

- 再调用Constructor对象的`newInstance()`方法创建该Class对象对应类的实例

- ```java
  Class<?> clazz = Class.forName(clazzName);//获取Class对象
  clazz.getConstructor().newInstance();//获取构造器后，生成对应类的对象
  ```

#### 调用方法

通过反射调用方法步骤：

- 先使用Class对象的`getMethod()`获取全部方法或指定方法，这个方法的返回值是Method数组，或者Method对象，每个Method对象对应一个方法。

- 通过Method对象调用`invoke()`方法

  - `Object invoke（Object obj, Object... args)`，该方法中的obj是执行该方法的的主调，args是传入方法的实参

- ```java
  Class<?> targetClass = target.getClass();
  Method mtd = targetClass.getMethod('setMoney', String.class);
  mtd.invoke(target, 100);
  ```

#### 操作成员变量

操作成员变量步骤

- 通过Class对象的`getField()`获取该类的所有成员变量或指定成员变量。

- Field对象提供了如下两组方法来读取或设置成员变量值

  - `getXxx(Object obj)`：获取obj对象的该成员变量的值。此处的Xxx对应8种基本类型，如果成员变量的类型为引用类型，则无需Xxx
  - `setXxx(Object obj,Xxx val)`：描述如上

- ```java
  Person p = new Person()
  
  Class<Person> personClazz = Person.class;
  
  Field nameField = personClazz.getDeclaredField("name");
  nameField.setAccessible(true);//设置通过反射访问该成员变量时取消访问权限检查
  nameField.set(p, "qin");
  
  Field ageField = personClazz.getDeclaredField("age");
  ageField.setAccessible(true);
  ageField.setInt(p, 30);
  ```

- 

#### 操作数组

java.lang.reflect包下还有一个Array类，Array对象可以代表所有数组，程序可以通过使用Array来动态地创建数组，操作数组元素等

- `static Object newInstance(Class<?> componentType, int... length)`：创建一个具有指定元素类型、指定维度的新数组

- `static xxx getXxx(Object array, int index)`,

- `static void setXxx(Object array, int index)`,

- ```java
  Object arr = Array.newInstance(String.class, 10);
  Array.set(arr,5,"QIN NI HAO");
  Object hello = Array.get(arr,5);
  system.out.println(hello);
  ```

## 7.4 动态代理

使用反射可以生成jdk动态代理

在java的java.lang.reflect包下提供了一个Proxy类和一个InvocationHandler接口，通过使用这个类和接口可以生成JDK动态代理类或动态代理接口。

Proxy提供了如下两个静态方法来创建动态代理类和动态代理实例

- `static Class<?> getProxyClass(ClassLoader loader,Class<?>... interfaces)`：创建一个动态代理类所对应的Class对象，该代理类将实现interfaces所指定的多个接口。
- `static Object newProxyInstance(ClassLoader loader, Class<?>[] interfaces, InvocationHandler h)`：直接创建一个动态代理对象，该代理对象的实现类实现了interfaces所指定的系列接口，执行代理对象的每个方法都会被替换执行InvocationHandler对象的invoke方法。

```java

//切面编程
//用户各异的自定义接口
public interface UserdiyOperationInterface{
    void sayHello();
    void Eat();
}
//用户甲对自定义接口的实现
public class UserdiyOperation implements UserdiyOperationInterface{
    public void sayHello(){
        System.out.println("大家好，很高兴认识大家");
    }
    public void eat(){
        System.out.println("大家好，很荣幸和大家聚餐");
    }
}
//用户操作的公共部分
public class AOPAchieve{
    public void beforeExecution(){
        System.out.println("========特别代码执行之前=========");
    }
    public void afterExecution(){
        System.out.println("========特别代码执行之后=========");
    }
}
//invocationHandler的实现类，该实现类的invoke方法将会作为代理对象的方法实现
class MyInvocationHandler implements InvocationHandler{
    private Object target;
    public void setTarget(Object target){
        this.target = target;
    }
    public Object invoke(Object proxy, Method method, Object[] args) throws Exception{
        AOPAchieve aop = new AOPAchieve();
        aop.berforeExecution();
        Object specialSection =method.invoke(target, args);
        aop.afterExecution();
        return specialSection
    }
}
//代理对象的工厂，该对象专为指定的target生成动态代理实例
public class MyProxyFactory{
    public static Object getProxy(Object target) throws Exception{
        MyInvocationHandler handler = new MyInvocationHander();
        handler.setTarget(target);
        return Proxy.newProxyInstance(target.getClass().getClassLoader(), target.getClass().getInterfaces(), handler);
    }
}

public class Test{
    public static void main(String[] args) throws Exception{
        //生成需要代理的对象
        UserdiyOperationInterface target = new UserdiyOperation();
        //生成代理实例
        UserdiyOperationInterface userdiyOperationInterface = (UserdiyOperationInterface) MyProxyFactory.getProxy(target);
        //代理后的方法执行
        userdiyOperationInterface.sayHello();
        userdiyOperationInterface.eat();
    }
}
```



## 7.5 反射与泛型

java允许泛型来限制Class类，eg：`String.class 的类型实际是 Class<String>。如果Class`对应的类型未知，则使用Class<?>

通过在反射中使用泛型，可以避免使用反射生成对象需要强制类型转换

# 8 多线程

引入进程的目的是为了更好地使多道程序并发执行，提高资源利用率和系统吞吐量。

引入线程的目的则是为了减少程序在并发执行是所付出的时空开销，提高操作系统的并发性能。

进程是除cpu外的系统资源的分配单元，而线程则作为CPU的分配单元。

线程之间共享进程的系统资源，为进程分配资源的时空开销大于线程分配资源的时空开销。

java使用Thread类代表线程，所有的线程对象都必须是Thread类或其子类的实例。

每个线程的作用是完成一定的任务，实际上就是执行一段程序流（一段顺序执行的代码）。java中使用线程执行体来代表这段程序流。

## 8.1 线程的创建和启动

### 8.1.2 继承Thread类创建线程

通过继承Thread类来创建并启动多线程步骤：

1. `定义Thread类的子类，并重写该类的`run()`方法，该run方法的方法体就代表了线程需要完成的任务。因此把`run()`方法称为线程执行体
2. 创建Thread子类的实例，即创建了线程对象
3. 调用线程对象的`start()`方法来启动该线程

```java
public class MyFirstThread extends Thread{
    private int i;
    public void run(){
        for(;i<100;i++){
            //要获取当前的对象，直接使用this
            System.out.println( getName() + "" + i );
            
        }
    }
    public static void main(String[] args){
        for(int i = 0; i < 100; i++){
            
            System.out.println(Thread.currentThread().getName() + "" + i); // main 46; main 47; main 48
            
            if(i == 20){
                new MyFirstThread().start();// Thread-0 24
                new MyFirstThread().start();// Thread-1 28
            }
        }
    }
}
```

此段代码开启了三个线程，包含两个显式创建的`Thred-0，Thread-1`，和一个主线程`main`。当java程序运行起来后，程序至少会创建一个主线程。

上面的代码中，用到了线程的两个方法：

- `Thread.currentThread()`：`currentThread()`是Thread类的静态方法，该方法总是返回当前正在执行的线程对象
- `getName()`：该方法是Thread类的实例方法，该方法返回调用该方法的线程名字
- `setName(String name)`：可以为线程设置名字

### 8.1.2 实现Runable接口创建线程类

步骤：

1. 定义`Runable`接口的实现类，并重写该接口的`run()`方法，该run方法的方法体同样是该线程的线程执行体
2. 创建Runable实现类的实例，并以此实例作为Thread的target来创建Thread对象，该Thread对象才是真正的线程对象。
3. 调用线程对象的`start()`方法来启动该线程

```java
public class MyRunableThread implements Runable{
    private int i;
    public void run(){
        for(;i<100;i++){
            //要获取当前的对象必须使用Thread.currentThread()方法
            System.out.println( Thread.currentThread().getName() + "" + i );
            
        }
    }
    public static void main(String[] args){
        MyRunableThread mrt = new MyRunableThread();
        new Thread(mrt, "新线程1").start();
        new Thread(mrt, "新线程2").start();
        //程序所创建的Runable对象只是线程的target，而多个线程可以共享同一个target
        //也就是说，采用 Runable接口的方式创建的多个线程可以共享线程类（实际上应该是线程target类）的实例变量
    }
}

```

### 8.1.3 使用Callable 和 Future创建线程

java目前不能把所有方法都包装成线程执行体。

自java5开始，java提供了Callable接口，它提供了一个call方法作为线程的执行体，call方法可以声明抛出异常，并可以有返回值，但Callable接口并没有实现Runable接口，所以不能直接作为Thread的target。

java5提供了Future接口来代表Callable接口里call()方法的返回值，并为Future接口提供了一个FutureTask实现类，该实现类实现了Futrue接口和Runable接口。

Future接口定义了几个如下的公共方法，来控制它关联的Callable任务。

- `boolean cancel( boolean mayInterruptRunning )`：试图取消Future关联的Callable任务
- `get()`：返回Callable任务里call方法的返回值，调用该方法将导致程序阻塞，会等到子线程结束才能等到返回值。
- `get(long timeout, TimeUnit unit)`：返回Callable任务里call方法的返回值，在指定的timeout时间内未返回值，将会抛出TimeOutException异常
- `boolean isCancelled()`：如果在Callable任务正常完成前被取消，则返回true。
- `boolean isDone()`：如果Callable任务已完成，则返回true

Callable接口有泛型限制，Callable接口里的泛型形参类型和call方法返回值类型相同，而且Callable接口是函数式接口，可以通过lambda表达式创建Callable对象。

创建并启动有返回值的线程步骤：

1. 创建Callable实现类，并实现有返回值的线程执行体`call()`方法，并创建Callable实现类的实例。
2. 使用FutureTask类来包装Callable对象。该FutureTask对象封装了该Callable对象的call方法的返回值。
3. 使用FutureTask对象作为Thread对象的target创建并启动线程
4. 调用FutureTask对象的get方法来获得子线程执行结束后的返回值

```java
class MyCallableThread{
    public static void main(String[] args){
        
        //先使用Lambda表达式创建Callable<Integer>对象
        //再使用FutureTask来包装Callable对象
        FutureTask<Integer> task = new FutureTask<Integer> (
            (Callable<Integer>) ()->{
                for(int i = 0; i < 100; i++){
                    System.out.println(Thread.curentThread().getName() + "的循环变量i的值：" + i);
                }
                //call方法的返回值
                return i;
            }
        )
        
        for(int i =0; i<100; i++){
            System.out.println(Thread.currentThread().getName() + "的循环变量i的值：" + i );
            if(i==20){
                new Thread(task,"有返回值的线程").start();
            }
            try{
                System.out.println("子线程的返回值："+ task.get());
            }catch(Exception e){
                ex.printStackTrace();
            }
        }
        
        
    }
    
}
```

### 8.1.4 创建线程的三种方式对比

后两种：

- 线程类只是实现了Runable、Callable接口，还可以继承其他类
- 多个线程可以共享一个target对象，非常适合多个线程处理同一份资源的情况
- 编程稍显复杂，访问当前线程需要使用`Thread.currentThread()`方法

通常情况建议采用后两种

# Debug

1. 单步调试：
   - step into：单步执行，遇到子函数就进入并且继续单步执行
   - step over：单步执行，将子函数整个作为一步（不会进入子函数单步执行），在不存在子函数的时候，和step into 的效果相似
   - step  out：在程序正在子函数内部执行时，通过step out就可以从执行完子函数剩余部分，返回上一层函数
2. 

