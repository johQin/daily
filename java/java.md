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
        System.out.println('hi,java')
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
    
    只有三种类型，基本类型，字符串类型和 null类型（只能赋值给引用类型的变量）

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

<h5>一、转型</h5>

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

java集合大致可分为Set、List、Queue和Map四种体系，其中Set代表无序、不可重复的集合；List代表有序重复的集合；而Map则代表具有映射关系的集合，java5又增加了Queue体系集合代表一种队列集合实现。

java集合就像一种容器，可以把多个对象（实际上是对象的引用，但习惯上都称对象）“丢进”该容器。

数组只能将统一类型的对象放在一起，而集合可以放多种类型的对象。

![IteratorTree.png](/legend/IteratorTree.png)

java集合类主要由两个接口派生而出：Collection和Map，

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

## 5.2 Set

Set集合通常不能记住元素的添加顺序，不允许包含相同的元素。有HashSet，TreeSet和EnumSet三个实现类。

### 5.2.1 HashSet

HashSet不是同步的，线程不安全。

当向HashSet集合中存入一个元素时，HashSet会调用该对象的hashCode方法来得到该对象的hashCode值，然后根据该hashCode值决定该对象在HashSet的存储位置。

HashSet集合判断两个对象相等的标准是两个对象通过equals()方法比较相等，并且两个对象的hashCode()方法返回也相等。

## 5.3 List

List集合代表一个元素有序（元素都有顺序索引，默认按照添加顺序设置元素索引）、可重复的集合。

List集合中增加了一些根据索引来操作集合元素的方法。

- void add(int index,Object o)
- boolean addAll(int index,Collection c)
- Objcet get(int index)
- int indexOf(Object o)
- int lastIndexOf(Object o)
- Object remove(int index)
- Object set(int index,Object element)
- List subList(int fromIndex,int toIndex)
- void sort( Comparator c )
- void replaceAll()

ArrayList 和 Vector作为List类的两个典型实现，都是基于数组实现的List类

## 5.4 Queue

## 5.5 Map

Map用于保存具有映射关系的数据（key-value对）

# 6 泛型

一旦把一个对象丢进java集合中，集合就会忘记对象的类型，把所有对象当成Object类型处理，当程序从集合中取出对象后，就需要进行强制类型转换，这种转换不仅使代码臃肿，而且容易引起classCastException异常。

## 6.1 使用泛型

```java
List<String> strList=new ArrayList<String>()
    //在集合接口或集合接口的实现类后增加尖括号，尖括号里放一个数据类型，即表明这个集合接口、集合类只能保存特定类型的对象。
Map<String,Integer> scores= new HashMap<String , Integer>();
```

## 6.2 java7 增强的"菱形语法"

```java
Map<String,Integer> scores= new HashMap<>();
//java7开始，允许在构造器后不需要带完整的泛型信息，只要给出一对尖括号，它可以推断括号里应该是什么泛型信息
```

## 6.3 深入泛型

所谓泛型，就是允许在定义类、接口、方法时使用类型传参，这个类型形参（或叫泛型）将在声明变量、创建对象、调用方法时动态地指定（即传入实际的类型参数，也可称为类型实参）、

```java
//定义接口时，指定了一个泛型形参，该形参名为E
public interface List<E>{
    //在该接口中，E可以作为类型使用
    void add(E x);
}
```

例如在使用List类型时，如果为E形参传入String类型实参，则产生了一个新的类型：List\<String>

泛型类也可以派生子类

# Debug

1. 单步调试：
   - step into：单步执行，遇到子函数就进入并且继续单步执行
   - step over：单步执行，将子函数整个作为一步（不会进入子函数单步执行），在不存在子函数的时候，和step into 的效果相似
   - step  out：在程序正在子函数内部执行时，通过step out就可以从执行完子函数剩余部分，返回上一层函数
2. 

