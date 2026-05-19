# C#

C# 编程是基于 C 和 C++ 编程语言的

C# 是由 Anders Hejlsberg团队（Microsoft）在 .Net 框架开发期间开发的。他属于.net框架的一部分。

虽然 C# 的构想十分接近于传统高级语言 C 和 C++，是一门面向对象的编程语言，但是它与 Java 非常相似。

# 0 绪论

## 0.1 .Net 框架

.Net 框架应用程序是多平台的应用程序，适用于多中语言：C#、C++，VB等。

.Net 框架由一个巨大的代码库组成，用于 C# 等客户端语言。

## 0.2 hello world

```c#
// 1. 引用命名空间（使用系统基础功能）
using System;

// 2. 自定义命名空间（代码的"文件夹"）
namespace HelloWorldDemo
{
    // 3. 类（程序的核心容器）
    class Program
    {
        // 4. Main 方法：程序入口（C# 程序从这里开始运行）
        static void Main(string[] args)
        {
            // 5. 程序语句（具体要执行的代码）
            Console.WriteLine("Hello, World!"); // 输出文字到控制台
        }
    }
}
```

1. `using System;` —— 引用命名空间
   - 作用：**引入系统自带的功能库**，就像 “提前告诉程序要用哪些工具”
   - `System` 是 C# 最核心的命名空间，包含控制台输入输出、字符串、数学等基础功能
   - 没有这行，后面写 `Console.WriteLine()` 会报错
2. `namespace HelloWorldDemo` —— 自定义命名空间
   - 作用：**给代码分组**，避免不同代码重名，相当于代码的 “文件夹”
   - 一个项目可以有很多命名空间，方便管理大量代码
   - 大括号 `{ }` 里的所有代码，都属于这个命名空间
3. `class Program` —— 类
   - C# 是**面向对象语言**，所有代码必须写在**类（class）** 里
   - 类是代码的 “容器”，方法、变量都要放在类中
   - 这里的 `Program` 是类名，可以自定义（比如改成 `MyApp`）
4. `static void Main(string[] args)` —— 程序入口（最重要）
   - **这是 C# 程序的唯一入口**，程序运行时**第一行执行的就是这里**
   - 每个类都可以有Main方法，但整个程序的入口在编译时必须指定并唯一。
   - 固定规则：
     - `static`：静态方法，不需要创建对象就能直接运行
     - `void`：这个方法没有返回值
     - `Main`：**必须叫这个名字**，大小写不能错
     - `string[] args`：可以接收命令行参数（简单程序可以不用）
5. `Console.WriteLine("Hello, World!");` —— 执行语句
   - 具体功能代码：在控制台输出一行文字
   - 每一条语句结尾必须加 `;` 分号（C# 语法强制要求）
   - `Console` 是系统提供的控制台类，`WriteLine` 是输出方法

# 1 变量

## 1.1 数据类型

在 C# 中，变量分为以下几种类型：

- 值类型（Value types）：存储在栈，直接存值，赋值拷贝副本

  - 整数：byte（无符号）， sbyte（有符号），short，ushort（无符号），int，uint，long，ulong

  - 浮点数：float，double，decimal

  - 布尔：bool

  - 字符类型：char，只能用单引号，并且只能是一个字符

  - 结构体：struct

  - 枚举：enum

  - ```c#
    sbyte b = -5;
    long f = 9999999999;
    
    // 浮点
    float f = 3.14f;	// 必须加f
    double d = 3.1415926;	// 默认的小数类型
    decimal money = 99.99m; // 钱用decimal，避免精度丢失
    
    bool isOk = true;
    bool isNo = false;
    
    char c = 'A';
    char c2 = '中'; // 支持中文
    
    struct Books
    {
       public string title;
       public string author;
       public string subject;
       public int book_id;
    };  
    
    
    // 枚举
    enum Days { Sun, Mon, tue, Wed, thu, Fri, Sat };
    ```

- 引用类型（Reference types）：存储在堆，存地址，赋值指向同一个对象

  - 字符串：String，可以通过两种形式进行分配：双引号和 @引号。@ 字符串中可以任意换行，换行符及缩进空格都计算在字符串长度之内。

  - 基类（万能类，所有类型的父类）：object

  - 数组：type[]

  - 类：class

  - ```c#
    string str = "Hello C#";	// 双引号，不可变
    string name = "张三";
    
    object obj1 = 100;
    object obj2 = "测试";
    
    int[] arr = {1,2,3};
    string[] names = {"小明","小红"};
    
    class Person{}
    Person p = new Person();
    ```

  

- 指针类型（Pointer types）

  - 日常开发 95% 场景几乎不用指针，C# 指针是小众、高级用法，一般只有底层 / 性能场景才用。
  - 工作 3–5 年做底层 / 游戏 / 高性能开发，再学指针

- 特殊类型

  - 可空类型：type?

  - var隐式类型（编译器自动推断）

    - 声明时必须赋值，不能改类型
    - 不能直接 `var x = null;`；`var` 能不能赋值 `null`，**看推断出来的类型支不支持 null**
    - 引用类型 / 可空类型可以后期赋值为 null

  - ```c#
    int? age = null; // int默认不能为null，加?就可以
    bool? b = null;
    
    var num = 10;    // 自动识别int
    var str = "abc"; // 自动识别string
    
    // 错误：无法推断var的类型，null没有默认类型
    var a = null; 
    
    // 给可空 / 引用类型赋值 null（编译器能识别）
    var str = (string?)null;   // 推断为 string?
    var num = (int?)null;      // 推断为 int?
    // 先给初始非 null，再赋值 null（最常用）
    var name = "张三";
    name = null;   // ✔ 合法，name是string类型，可以为null
    ```

  - 

###  类型转换

类型转换是将一个数据类型的值转换为另一个数据类型的过程。

C# 中的类型转换可以分为两种：

- **隐式类型转换**：指将一个较小范围的数据类型转换为较大范围的数据类型时，编译器会自动完成类型转换

- **显式类型转换**（也称为强制类型转换）：一个较大范围的数据类型转换为较小范围的数据类型时，或者将一个对象类型转换为另一个对象类型时，需要使用强制类型转换符号进行显示转换，强制转换会造成数据丢失。

```c#
int i = 10;
byte b = (byte)i; // 显式转换，需要使用强制类型转换符号

double doubleValue = 3.14;
int intValue = (int)doubleValue; // 强制从 double 到 int，数据可能损失小数部分
```

c# 包含一些内置的类型转换方法

## 1.2 变量定义

C# 4.0引入了动态类型 (dynamic)，它允许在运行时推断变量的类型。这在一些特殊情况下很有用，但通常最好使用静态类型以获得更好的性能和编译时类型检查。

**变量命名规则**

- 变量名可以包含字母、数字和下划线。
- 变量名必须以字母或下划线开头。
- 变量名区分大小写。
- 避免使用 C# 的关键字作为变量名。

```c#
<data_type> <variable_list>;

int i, j, k;
char c, ch;
float f, salary;
double d;

int d = 3, f = 5;    /* 初始化 d 和 f. */
byte z = 22;         /* 初始化 z. */
double pi = 3.14159; /* 声明 pi 的近似值 */
char x = 'x';        /* 变量 x 的值为 'x' */
```

## 1.3 作用域

变量的作用域通常由花括号 **{}** 定义的代码块来确定。

## 1.4 运算符

和java基本相似

### 成员访问运算符**`?.`**

```c#
class Person
{
    public string Name { get; set; }
    public Address Address { get; set; }
}

class Address
{
    public string City { get; set; }
}

// 安全访问属性
Person person = null;
string cityName = person?.Address?.City;  // 返回 null，不会抛出异常
```

### 元素访问运算符 `?[]`

```c#
int[] numbers = null;
int? firstNumber = numbers?[0];  // 返回 null，不会抛出异常

numbers = new int[] { 1, 2, 3 };
firstNumber = numbers?[0];  // 返回 1
```

| 运算符 | 名称                | 用途                      | 示例             | 返回值         |
| :----: | :------------------ | :------------------------ | :--------------- | :------------- |
|  `?`   | 可空类型声明        | 声明可以为 null 的值类型  | `int? x = null;` | null           |
|  `??`  | Null 合并运算符     | 如果左侧为 null，返回右侧 | `a ?? b`         | b              |
|  `?.`  | Null 条件运算符     | 安全访问成员              | `a?.b`           | null 或 b 的值 |
| `?[]`  | Null 条件索引运算符 | 安全访问数组或集合        | `a?[i]`          | null 或元素    |
|  `?:`  | 条件（三元）运算符  | 根据条件返回不同值        | `a ? b : c`      | b 或 c         |

## 1.5 判断语句

```c#
if(boolean_expression)
{
   /* 如果布尔表达式为真将执行的语句 */
}
else
{
  /* 如果布尔表达式为假将执行的语句 */
}


switch(expression){
    case constant-expression  :
       statement(s);
       break; 
    case constant-expression  :
       statement(s);
       break; 
  
    /* 您可以有任意数量的 case 语句 */
    default : /* 可选的 */
       statement(s);
       break; 
}

```

## 1.6 循环语句

```c#
while (a < 20)
{
    Console.WriteLine("a 的值： {0}", a);
    a++;
}

for (int a = 10; a < 20; a = a + 1)
{
    Console.WriteLine("a 的值： {0}", a);
}

int[] fibarray = new int[] { 0, 1, 1, 2, 3, 5, 8, 13 };
foreach (int element in fibarray)
{
    System.Console.WriteLine(element);
}

do
{
   Console.WriteLine("a 的值： {0}", a);
    a = a + 1;
} while (a < 20);
```

break：终止 **loop** 或 **switch** 语句，程序流将继续执行紧接着 loop 或 switch 的下一条语句。

continue：跳过本轮循环，开始下一轮循环。

# 2 面向对象

## 2.1 访问修饰符

*一个* **访问修饰符** *定义了一个类成员的范围和可见性。*

| 修饰符                 | 含义             | 访问范围（通俗版）                     |
| ---------------------- | ---------------- | -------------------------------------- |
| **public**             | 公开             | 任何地方都能访问（项目内、其他项目）   |
| **protected**          | 受保护           | 自己类 + 子类（继承的类）              |
| **internal**           | 内部             | **当前整个项目**内都能用，其他项目不行 |
| **protected internal** | 内部受保护       | 当前项目内 + 子类（跨项目子类也可）    |
| **private**            | 私有（**默认**） | **只能自己类内部**，外面完全看不到     |

`internal` 修饰符里说的**当前项目**，就是一个 `.csproj` 文件对应的工程。

一个项目编译后 = 1 个 `.dll` 或 `.exe` = 1 个程序集

`internal` = **同一个程序集内可见**

## 2.2 方法

```c#
<访问修饰符> <返回类型> <方法名>(<参数列表>)
{
    方法体
}

using System;

namespace MethodDemo
{
    class NumberHelper
    {
        // 定义方法：接收两个 int，返回较大的那个
        public int FindMax(int num1, int num2)
        {
            if (num1 > num2)
                return num1;
            else
                return num2;
        }

        static void Main(string[] args)
        {
            // 创建对象并调用方法
            NumberHelper helper = new NumberHelper();
            int result = helper.FindMax(100, 200);

            Console.WriteLine($"最大值是：{result}");  // 最大值是：200
        }
    }
}
```

### **参数传递方式**

- 值传递：默认传递方式，实参的值会被**复制**给形参。

- 引用传递：使用 `ref` 关键字传递变量的**引用**（内存地址），方法内对形参的修改会直接影响原始变量。注意调用时也必须加上 `ref`

  - ```c#
    using System;
    
    namespace ParameterDemo
    {
        class Program
        {
            // 引用传递：x 和 y 直接引用 a、b 的内存
            public void Swap(ref int x, ref int y)
            {
                int temp = x;
                x = y;
                y = temp;
            }
    
            static void Main(string[] args)
            {
                var p = new Program();
                int a = 100, b = 200;
    
                Console.WriteLine($"调用前：a={a}, b={b}");
                p.Swap(ref a, ref b);  // 调用时也必须加 ref
                Console.WriteLine($"调用后：a={a}, b={b}");  // 交换成功！
            }
        }
    }
    ```

- 输出参数传递：使用`out`关键字传递变量的**引用**，方法内对形参的修改会直接影响原始变量。在调用函数时，实参无需初始化，但函数内必须为其赋值。

  - ```c#
    using System;
    
    namespace ParameterDemo
    {
        class Program
        {
            // out 参数：方法负责赋值
            public void GetValues(out int x, out int y)
            {
                Console.Write("请输入第一个值：");
                x = Convert.ToInt32(Console.ReadLine());
    			Console.Write("请输入第二个值：");
                y = Convert.ToInt32(Console.ReadLine());
                // 注意：方法结束前必须为所有 out 参数赋值，否则编译错误
            }
    
            // 实用示例：TryParse 模式
            public bool TryDivide(int a, int b, out double result)
            {
                if (b == 0)
                {
                    result = 0;
                    return false;  // 除数为零
                }
                result = (double)a / b;
                return true;
            }
    
            static void Main(string[] args)
            {
                var p = new Program();
    
                // out 参数无需初始化
                int a, b;
                p.GetValues(out a, out b);
                Console.WriteLine($"a={a}, b={b}");
    
                // TryParse 模式
                if (p.TryDivide(10, 3, out double res))
                    Console.WriteLine($"10 / 3 = {res:F2}");  // 3.33
            }
    	}
    }
    ```



### 默认参数与命名参数

C# 允许为参数设置**默认值**，调用时可以省略这些参数。配合**命名参数**，还可以跳过某些参数只指定后面的：

```c#
using System;

namespace MethodFeatures
{
    class Program
    {
        // 默认参数：power 默认为 2
        static double Power(double baseNum, int power = 2)
        {
            double result = 1;
            for (int i = 0; i < power; i++)
                result *= baseNum;
            return result;
        }

        // 多个默认参数
        static void PrintInfo(string name, int age = 18, string city = "未知")
        {
            Console.WriteLine($"{name}, {age}岁, {city}");
        }

        static void Main(string[] args)
        {
            // 使用默认参数
            Console.WriteLine(Power(3));       // 9.0（使用默认 power=2）
            Console.WriteLine(Power(3, 3));    // 27.0

            // 命名参数：跳过 age，只指定 city
            PrintInfo("小明", city: "北京");
            // 输出：小明, 18岁, 北京

            // 命名参数可以不按顺序
            PrintInfo(city: "上海", name: "小红", age: 25);
            // 输出：小红, 25岁, 上海
        }
    }
}
```

### 方法重载

同一个类中，方法名相同，参数列表不同，就是方法重载；返回值不能区分重载。

访问修饰符可以不一样，不影响重载

## 2.3 构造与析构函数

构造函数的名称与类的名称完全相同，它没有任何返回类型。可以重载。默认有一个无参构造函数。

析构函数的名称是在类的名称前加上一个波浪形（~）作为前缀，它不返回值，也不带任何参数。

调用顺序：子类静态 → 父类静态 → 父类实例构造 → 子类实例构造 → 子类析构 → 父类析构

## 2.4 继承

```c#
class <派生类> : <基类>
{
 ...
}
```

派生类继承了基类的成员变量和成员方法。因此父类对象应在子类对象创建之前被创建。	

# 3 高级

## 3.1 反射

反射是**运行时动态发现类、方法、特性、属性**。

反射（Reflection）有下列用途：

- 它允许在运行时查看特性（attribute）信息。
- 它允许审查集合中的各种类型，以及实例化这些类型。
- 它允许延迟绑定的方法和属性（property）。
- 它允许在运行时创建新类型，然后使用这些类型执行一些任务。

### 3.1.1 特性

特性是 C# 中**给代码（类、方法、属性、参数等）附加额外信息**的一种机制，它本质是一个**继承自 `Attribute` 的类**，可以在**编译时 / 运行时**通过反射**读取**这些附加信息，用来实现配置、校验、标记、框架功能等。

基础语法：

1. 特性用 **`[ ]`** 包裹，放在要修饰的代码上方 / 前方
2. 特性类名通常以 `Attribute` 结尾，使用时可以省略
3. 支持位置参数、命名参数

#### 自定义特性

必须继承 `System.Attribute`，推荐加上 `AttributeUsage` 限定使用范围。

```c#
// 自定义特性
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method)]
public class AuthorAttribute : Attribute
{
    public string Name { get; set; }
    public string Version { get; set; }

    public AuthorAttribute(string name)
    {
        Name = name;
    }
}

// 类（标记：张三）
[Author("张三", Version = "1.0.0")]
public class OrderService
{
    // 方法（标记：李四）
    [Author("李四")]
    public void CreateOrder()
    {
        Console.WriteLine("执行 CreateOrder 方法");
    }
}



class Program
{
    static void Main()
    {
        // 1. 创建实例并调用方法（只会执行方法，不读特性）
        OrderService service = new OrderService();
        service.CreateOrder(); // 输出：执行 CreateOrder 方法

        // 2. 反射读取【类】的特性 → 张三
        Type type = typeof(OrderService);
        var classAttr = Attribute.GetCustomAttribute(type, typeof(AuthorAttribute)) as AuthorAttribute;
        Console.WriteLine("类作者：" + classAttr?.Name); // 张三

        // 3. 反射读取【方法】的特性 → 李四
        var method = type.GetMethod("CreateOrder");
        var methodAttr = Attribute.GetCustomAttribute(method, typeof(AuthorAttribute)) as AuthorAttribute;
        Console.WriteLine("方法作者：" + methodAttr?.Name); // 李四
    }
}
```



### 3.1.2 使用反射的场景

场景：权限校验（最经典、最常用）

系统里有很多方法，有的需要**管理员才能调用**，有的普通用户就能调用。

不想每个方法都写一堆 `if(role!="admin") return;`，太啰嗦。

```c#
// 自定义特性
// 标记这个方法需要什么角色
[AttributeUsage(AttributeTargets.Method)]
public class NeedRoleAttribute : Attribute
{
    public string Role { get; }
    public NeedRoleAttribute(string role)
    {
        Role = role;
    }
}

// 修饰业务类
public class UserService
{
    // 普通用户可访问
    public void GetUserInfo()
    {
        Console.WriteLine("查询用户信息");
    }

    // 必须管理员
    [NeedRole("Admin")]
    public void DeleteUser()
    {
        Console.WriteLine("删除用户");
    }
}


public static void InvokeMethod(object obj, string methodName, string currentUserRole)
{
    Type type = obj.GetType();
    MethodInfo method = type.GetMethod(methodName);

    // ====== 反射读取方法上的特性 ======
    var needRole = method.GetCustomAttribute<NeedRoleAttribute>();

    if (needRole != null)
    {
        // 如果需要管理员，但当前角色不是 → 拒绝执行
        if (needRole.Role != currentUserRole)
        {
            Console.WriteLine("权限不足，禁止执行！");
            return;
        }
    }

    // 权限通过，执行方法
    method.Invoke(obj, null);
}

var service = new UserService();

// 当前用户是普通用户
InvokeMethod(service, "DeleteUser", "User");  
// 输出：权限不足，禁止执行！

// 当前用户是管理员
InvokeMethod(service, "DeleteUser", "Admin"); 
// 输出：删除用户
