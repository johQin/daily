# Python

# 1 基础

- python是一种面向对象、解释型、弱类型的脚本语言

- python程序严格区分大小写，
- python程序不要求语句使用分号结尾，当然也可以使用分号

## 1.1 注释

- 单行注释：**#**，跟在#后面直到这行结束的代码都将被注释
- 多行注释：三个单引号**'''**和三个双引号**"""**，可以将多行的代码注释掉
- 退出python命令行：【Ctrl +z】或【quit() + Enter】或【exit() + Enter】

## 1.2 变量

python是**弱类型语言**，特点：

- 变量无须声明即可直接赋值
- 变量的数据类型可以动态改变，可以被赋不同类型的值

### 1.2.1 打印输出函数print()

<pre>print(val1,val2,...,valn,[sep=分割符,end=结束符,file=输出目标,flush=输出缓存控制])</pre>

### 1.2.2  变量命名规则

**python语言的标识符必须以字母、下划线（_）开头，后面可以接任意数目的字母（包括中文，日文，英文），数字和下划线**

### 1.2.3 关键字和内置函数

关键字：

| false | none   | ture     | and   | as     | assert   |
| ----- | ------ | -------- | ----- | ------ | -------- |
| break | class  | continue | def   | del    | elif     |
| else  | except | finally  | for   | from   | global   |
| if    | import | in       | is    | lambda | nonlocal |
| not   | or     | pass     | raise | return | try      |
| while | with   | yield    |       |        |          |

内置函数：

许多，

## 1.3 数值类型

为提高数值的可读性，python 3.x允许为数值增加下划线作为分隔符而并不会影响数值本身。

整型：十进制/二进制（0b开头）/八进制（0o开头）/十六进制（0x开头），eg：26/0b11010/0o32/0x1a

浮点数：普通浮点数/科学计数，eg：3.1415926E3

复数：python支持复数，复数的虚部用j或J表示，eg：3+0.2j，复数的计算模块 cmath。使用时，必须导入：import cmath

## 1.4 字符串

python要求字符串必须使用引号（单引号或双引号）括起来。支持**转义字符（\\)**

### 1.4.1 基本使用

**拼接**

1. 字符串与字符串：加号（+）拼接，s4 = s1 + s2
2. 字符串与数字：python不允许直接拼接数值和字符串，必须先将数值转换成字符串（  str(num)   、repr(  num  )  ），然后再用加号连接两个字符串

**字符串输入**

input()函数用于向用户生成一条提示，然后获取用户输入的内容，并返回一个字符串。

```python
msg=input("请输入你的值：");
print(type(msg));
print(msg);
```

**长字符串**

用多行注释的方法，将注释的内容赋值给变量，这就是长字符串，他可以换行书写便于查看，

**字节串**

bytes保存的就是原始的字节（二进制格式）数据，因此bytes对象可用于在网络上传输数据，也用于存储各种二进制格式文件，如图片，音乐等文件。

一个汉字三个字节。有三种方式将字符串转为bytes对象。不做详述。

### 1.4.2 深入使用

1. 字符串格式化输出

```python
print("%s is a %s years old boy" % (user,age));#  %后面是转换说明符，格式输出类似于C语言的格式输出
```

2. 索引操作字符串（<b> [  ] </b>）

3. **in**判断是否包含某个子字符串

4. **字符串方法len()，strip()，find()，replace()，maketrans()等等，用时可查**

## 1.5 运算符

1. 赋值运算符：支持连续赋值，**=**，扩展赋值运算符

2. 算术运算符：+ ， -  ， *，  /（普通除），  //（整除），  % ，**\*\*（乘方）** 

3. 位运算符

4. 索引运算符**[  ]**

5. 比较运算符，**判断引用对象is， is not**

6. 逻辑运算符，**and，or，not**

7. in运算符，用于判断某个成员是否位于序列中

   <h5>三目运算符</h5>

   <pre>
   	True_statement if 条件表达式 else False_statement    
   </pre>

## 1.5 帮助文档

python非常方便，它甚至不需要用户查询文档，python是自带文档的，

- **dir(  )**：列出指定类或模块包含的全部内容（包括函数、方法、类、变量等）
- **help(  ) :**查看某个函数或方法的帮助文档

<pre>
  eg：dir(  str )，将会列出str类提供的所有方法。
  help(str.title)，列出str类的title方法
</pre>
# 2 python常用数据结构

Python内置三种常用的三种数据结构：列表（list）、元组（tuple）和字典（dict）

在python中，语句中包含**[  ]，{  }，(  ) **括号中间换行的就不需要使用多行连接符。

## 2.1 序列

python的常见序列类型包括字符串（不可变）、元组（不可变）和列表（可变）等;

### 2.1.1 创建元组和列表

<pre>
  元组：(ele1,ele2,...,elen)
  列表：[ele1,ele2,...,elen]
</pre>

### 2.1.2 元组和列表的公共用法

1. 索引使用：**[  +/-  n ]**

2. 子序列:**[ start(起始索引，包含) : end(结束索引，不包含)  : step(步长)  ]**

3. 加法：列表只能加列表，元组只能加元组，两个序列首尾衔接

4. 乘法：n个序列首尾衔接，还可以混合乘法和加法运算

5. in：判断是否包含某个元素

6. 序列封包和解包

   序列支持以下两种赋值方式

   - 程序把多个值赋值给一个变量时，系统会自动将多个字封装成元组——序列封包
   - 程序允许将序列（元组或列表）直接赋值给多个变量，要求：变量数和序列元素个数相同——序列解包
     - 解包时，python允许被赋值的变量之前添加**“ * ” **，那么该变量就代表一个列表，可以保存集合中多个元素

   ```python
   #索引用法
   a_tuple=("autumn_moon",10,2.6);
   print(a_tuple[0]);# autumn_moon
   print(a_tuple[-1]);# 2.6
   #子序列
   a_list=[15,"ordinary_world",9.5];
   print(a_list[0,-2])#[15,"ordinary_world"]
   #加法
   b_list=[10,"extraordinary_person",12.5];
   sum_list=a_list+b_list;
   print(sum_list);#[15,"ordinary_world",9.5,10,"extraordinary_person",12.5]
   #乘法
   mul_list=a_list*3;
   print(mul_list);#[15,"ordinary_world",9.5,15,"ordinary_world",9.5,15,"ordinary_world",9.5]
   #in
   print(15 in a_list);#true
   piint(15 not in a_list);#false
   #序列封包
   pack=1,2,3;
   print(pack);#(1,2,3)
   #序列解包
   a,b,c,d=(1,2,3,4);
   #星号*的使用，和ES6中剩余参数相似
   j,*k,l=range(6)	#j=1,k=[2,3,4,5],l=6
   #封解包混合使用
   e,f,g,h=1,2,3,4
   
   ```

   

## 2.2 列表

1. 创建列表：

   - **[  ]**：直接创建

   - **list(  )**：可用于将元组，区间（range）等对象转换为列表

2. 增：
   - append()：追加参数到列表最后面（让参数成为一个元素），参数可以为单个值，元组，列表
   - extend()：也是追加，当参数是序列时，拆分成多个元素依次追加。消除嵌套的情形

3. 删：del、索引、子序列和赋空列表的方法可删除，

4. 改：索引，子序列

   ```python
   a_list=[1,2,3]
   a_tuple=(5,6,7)
   #元组转列表
   b_list=list(a_tuple)
   #增
   a_list.append(4)#[1,2,3,4]
   a_list.append(b_list)#[1,2,3,4,[5,6,7]]
   c_list=[8,9,10]
   c_list.extend(b_list)#[8,9,10,5,6,7]
   #删
   del a_list[4]#[1,2,3,4]
   del a_list[0:2:2]#[2,4]
   c_list[1:2]=[];#[8,5,6,7]
   #插
   a_list[1:1]=['a','b']
   print(a_list)#[2,'a','b',4]
   ```


5. 其他方法
   - count()，统计列表中某元素的出现次数
   - index()，判读列表中某元素的位置
   - pop()，出栈
   - reverse()，倒序
   - sort()，排序

## 2.3 字典

1. 创建字典
   - 花括号语法：**{k1:v1,k2:v2,...,kn:vn}**
   - dict()函数：传入映射键值对**dict([k1,k2,k3],[v1,v2,v3]) ** /   传入列表（里的每一个元素是一个只包含键和值两个元素的元组或列表，eg：**dict([[k:v],[k1:v1],(k2,v2)])**）

2. 访问：通过key访问value使用的也是方括号语法
3. 增：通过对不存在的key赋值
4. 改：对存在的key赋新值
5. 删：通过del删

```python
#创建
a_dict={'chinese':150,'english':150,('biology','geography','history'):90}
b_dict=dict([['chinese',150],['english':150],['math':150]])
c_dict=dict(['chinese','english','math'],[150,150,150])
#访问
print(a_dict['chinese'])#150
#增
a_dict['math']=150
#改
a_dict['english']=100
#删
del a_dict[('biology','geography','history')]
```

6. 其他方法
   - clear()：清空字典，返回空字典{}
   - get()：无key则返None
   - update()：有则更，无则增
   - items(),keys(),values()：返回键值对，键，值的对象，可通过list做转换
   - pop(),popitem()
   - fromkeys()

7. 注意：字典相当于索引是任意不可变类型的列表，因此元组可以做字典的索引，而列表不能。

# 4 流程控制

## 4.1 代码块

python不是格式自由的语言，不像java语言的代码块是用花括号**{  }**区别，python的代码块是靠缩进来区别的

代码块一定要缩进，否则就不是代码块。**不要随意缩进，同一个代码块内的代码必须保持相同的缩进**。

通常引起代码块的语句都有一个**“ : ”**

**pass**语句：空语句，该语句不做任何事情，只做标记和预留位置的作用。

### 4.1.1 with 上下文管理

```python
with EXPR as VAR:
    BLOCK
```

基本思想是with所求值的对象必须有一个\_\_enter\_\_()方法，一个\_\_exit\_\_()方法。

紧跟with后面的语句被求值后，返回对象的\_\_enter\_\_()方法被调用，这个方法的返回值将被赋值给as后面的变量（as 语句可有可无）。当with后面的代码块全部被执行完之后，将调用前面返回对象的__exit__()方法。

```python
class Sample:
  def __enter__(self):
    print("In __enter__()")
    return "Foo"
  
  def __exit__(self, type, value, trace):
    print("In __exit__()")
  
def get_sample():
  return Sample()
  
with get_sample() as sample:
    print("sample:%s" % sample)
    ha = 5
print("ha = %s" % ha)
#ha = 5
#主程序段依然能访问到，with里面声明的变量
```

运行代码后，输出如下：

In \_\_enter\_\_()
sample: Foo
In \_\_exit\_\_()

## 4.2 条件结构

### 4.2.1 if结构

<pre>
  if expression :
  	statements...
  elif expression:
  	statements...
  ...
  else:
  	statement...
</pre>

if条件可以是任意类型

下面的值会被解释器当做**false**处理

False、None、0、“ ”、（ ），[ ]、{ }

### 4.2.2 断言

<pre>
  assert expression
</pre>

true程序继续向下执行，false会引发AssertionError错误

## 4.3 循环结构

循环控制：break（结束此循环）、continue（结束本次循环）

### 4.3.1 while循环

<pre>
  [init_statements]
  while condition_expresstion :
  	body_statements
  	[iteration_statements]
</pre>

初始化语句、循环条件、循环体、迭代语句

### 4.3.2 for-in循环

专门用于遍历范围、列表、元素和字典等可迭代对象包含的元素

<pre>
    for 变量  in  字符串 | 范围 | 集合 ：
    	statements
</pre>

变量的值受for-in循环控制，改变量将会在每次循环开始时自动被赋值，因此程序不应该对该变量赋值。

<h5>for-in遍历字典</h5>

字典包含items()，keys()，values()三个方法，他们都返回三个列表。

<h5>循环结束else</h5>

当循环条件为False时，程序会执行else代码块，它的主要作用是便于生成更优雅的Python代码。

当循环未被break终止时，程序会执行else块中的语句。

### 4.3.3 for表达式

for表达式用于利用其它区间、元组、列表等可迭代对象创建新的列表

<pre>
    [ 表达式 for 变量  in 可迭代对象 [ if cond_expression ] ]
</pre>

如果将**方括号[]换为圆括号()**，这样表达式不会再生成列表，而生成一个生成器（generator），他也可以for-in循环。

for表达式也可**嵌套循环**

### 4.3.4 常用工具函数

1. zip()，可以把多个列表（以长度最短的列表为标准）压缩成一个zip对象（可迭代对象），这个可迭代对象所包含的元素是由原列表元素组成的元组。
2. reversed()，可接收各种序列（元组，列表、区间等）的参数，返回反序排列的迭代器。
3. sorted(sequence,reverse)，默认从小到大排，reverse参数为true，就从大到小排列了。

```python
target={'语文' : 120,'数学':110,'英语':110}
for key,val in target.items():
    print("科目：%s \n 成绩：%s \n " % (key,val))
else :
	print('循环结束');
a_range=range(10)

#for语句
b_list=[x*x for x in a_range if x % 2 == 0]
print(b)#[0,4,16,36,64]

#for语句的嵌套循环
c_list=[(x,y) for x in range(3) for y range(4)]
print(c_list)#[(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3)]

#zip()
a=['a','b','c']
b=['1','2','3']
c=zip(a,b)		#c=[('a',1),('b',2),('c',3)]
```

# 5 函数

## 5.1 函数定义

<pre>
    def 函数名(param1,param2,...):
    	'''
    	函数说明文字，只要把一段字符串放在函数声明之后，函数体之前，这段字符串将作为函数的说明文档
    	'''
    	函数体
    	[return [返回值]]
</pre>

- 参数列表：一旦在定义函数时制定了形参列表，**调用该函数时就必须传入对应的参数值。**
- 函数说明文字：通过  **help(函数名)**或打印**print(函数名.\_\_doc\_\_)**，可以查看函数的说明文档。
- 返回值：如果程序需要有多个返回值，我们可以主动包装成列表之后返回。如果python函数直接返回多个值，**python会自动将多个返回值封装成元组**

```python
def max(x,y) :
	'''
	比较两个数，获取较大数的值。

	max(x,y)
	返回x和y数之间较大的那个数。
	'''
	z = x if x > y else y
	return z
```

## 5.2 函数参数

### 5.2.1 位置和关键字参数

- 位置参数：按照形参位置传入的参数，必须按照形参的顺序传入参数值。

- 关键字参数：按照形参名传入指定的参数值，无需按照形参定义的顺序传入参数值。

  **注意：位置参数必须在关键字参数之前。**

### 5.2.2 参数默认值

在定义函数时，可以为形参指定默认值，这样在调用函数时就可以省略为该形参传入参数值，而是直接使用该形参的默认值。

python要求将带默认值的参数定义在形参列表的最后。

```python
def girth(width,height=20):
    print('width:',width)
    print('height:',height)
    return 2*(width+height)
print(girth(10,12))#普通参数就是位置参数
print(girth(height=6,width=5))#关键字参数，可交换位置

print(girth(2.5,height=3))#位置参数只能在关键字参数之前

print(girth(10))#参数默认值，可省略有默认值的参数的传入
```



### 5.2.3 参数收集

和JavaScript的剩余参数相似。python允许在形参前面添加星号（  *  ），这样就意味着，该参数可接收多个参数值。

有两种参数收集方式

- 普通参数收集：形参名前面加一个星号（**"  *  " **），多个普通参数值被当成元组传入。
- 关键字参数收集：形参名前面加两个星号（**" * * "**)，多个以关键字参数传入的参数将被当成字典传入。

一个函数最多只能带一个支持"普通参数收集"的形参。

```python
def test(x,y,z=3,*books,**scores) :
    print(x,y,z)
    print(books)
    print(scores)
print(test(1,2,3,"高等数学","大学生英语"，语文=70,数学=110))
#1 2 3
#('高等数学','大学生英语')
#{语文:70,数学：110}
```



<h5>逆向参数收集</h5>

所谓逆向参数收集，指的是在程序已有列表，元组、字典等对象的前提下，把他们的元素**“拆开”**后传给函数的参数。

逆向参数收集需要在传入的列表、元组参数之前添加一个星号，在字典参数之前添加两个星号。

```python
def foo(name,*scores):
    print('学生姓名：',name)
    print('主科分数：',scores)
a_tuple=(110,120,115)
foo('秦奋',*a_tuple)
#学生姓名：秦奋
#主科分数: (110,120,115)

def bar(book,price,desc):
    print("书名：%s，价格：%s，描述：%s" % ( book,price,desc))
my_book={'price':34,'book':'数据科学与大数据分析','desc':'一本关于数据的书'}
bar(**my_book)
```

### 5.2.4 传参机制

python中函数的参数传递机制都是——值传递。

对象传入和java的相似，依旧不是传址。

如果函数收到的是一个可变对象（比如字典或者列表）的引用，就能修改对象的原始值－－相当于通过“传引用”来传递对象。如果函数收到的是一个不可变对象（比如数字、字符或者元组）的引用，就不能直接修改原始对象－－相当于通过“传值'来传递对象。

## 5.3 变量作用域

局部变量：在函数中定义的变量，包括参数。

全局变量：在函数外面，全局范围内定义的变量。

python提供如下三个工具来获取指定范围内的**“变量字典”**（键为变量名，变量值为对应键的值）

- globals()：该函数返回全局范围内所有变量组成的“变量字典”
- locals()：该函数返回当前局部范围内所有变量组成的“变量字典”
- vars( obj )：获取在指定对象obj范围下所有变量组成的“变量字典”，如果不传入obj参数和locals()的作用相同。

使用globals()和locals()获取的变量字典只应该被访问（读），不应该被修改（写）。但实际上是可以被修改的

全局范围内，locals和globals获取的变量字典可读可写。局部范围内，locals获取的变量字典只可读不可写，glabal依旧可读可写。

## 5.4 局部函数

在函数体内定义的函数——局部函数

局部函数只能在封闭的函数内有效

## 5.5 函数变量

函数变量——所有函数都是function对象，这意味着可以把函数本身赋值给变量。而被赋值的变量也可以通过函数调用的形式调用函数。

既然函数可以作为变量，那么**函数也可以作为函数的形参**。当实参传的是函数的时候，那么形参将接收到一个函数。

函数变量、形参和实参都不需要带**(  )**，只有在做函数使用的时候需要带。

## 5.6 lambda表达式

lambda 表达式的本质是匿名的、单行函数体的函数。

<pre>
    lambda [param1,param2,...] : 表达式
</pre>

好处：

- 对于单行函数，使用lambda表达式的适应性更强。
- 对于不需要多次复用的函数，使用lambda表达式可以在用完之后立即释放掉。
- lambda可以作为函数参数被传递，在三目运算符中也经常被用到，也经常用来替代局部函数。

## 5.7 函数装饰器@

使用@符号引用已有的函数，可用于修饰其他函数

使用“@函数A”去装饰函数B，实际上完成了如下两步

- 首先，将B作为参数传给A函数
- 然后，将B替换为A函数的返回值

因此，被修饰的函数B完全由A函数的返回值决定。

函数装饰器是一个非常实用的功能，它既可以在被修饰函数之前添加额外的处理逻辑，也可以在之后添加一些处理逻辑。

```python
def A(fn):
	def bar(param):
		print("参数是：",param)
		return fn(param*2)
	return bar
@A
def B(param):
	print("输入参数的平方：",param*param)
    return param*param
print(B)#<function A.<locals>.bar at 0x00000000021FA...>
#B函数实质上已被替换为bar函数
print(B(10))
#参数是：10
#输入参数的平方：400
#400
```

# 6 面向对象

## 6.1 类和对象

类是面向对象的重要内容，可以把类当成一种自定义类型，可以使用类来定义变量，也可以使用类来创建对象

### 6.1.1 类

<pre>
    class 类名：
    	执行语句...
    	类变量...
    	方法...
</pre>
1. 类变量——在类体中为新变量赋值就是增加类变量。通过del语句即可删除已有类变量。
2. 实例变量——只要对新实例变量进行赋值就是增加实例变量。通过del语句即可删除已有对象的实例变量。
3. 类中定义的方法默认是实例方法。方法的定义和函数的定义基本相同。只是**实例方法的第一个参数会被自动绑定到方法的实例对象——因此实例方法的第一个参数应该定义为self**
4. 构造方法\__init__( self , param1 , ...)。构造方法是一个类创建对象的根本途径。如果开发者没有显示定义此函数，那么python会自动为该类定义一个包含self的默认的构造函数。
5. python允许在类范围内放置可执行代码。

### 6.1.2 对象

创建对象的根本途径是调用构造方法，python无须像java通过new调用构造方法，也不需显示通过类显示调用\__init__(self,... )。仅仅通过

```python 
obj =className(self,init_params)
```

python对象大致有如下作用

- 操作对象的实例变量（增，删，改）
- 操作对象的方法（增，删，改）

```python
class Person :
    #类变量
    hair="black"
    
    #构造函数的self指向正在初始化的对象
    def __init__(self,name = 'john',age=22):
        #构造函数里可初始化对象的实例变量，需要在声明实例时，传入相应的参数，当然在这里已有默认值了
        self.name=name
        self.age=age
    def say(self,content):
    	print(content)
   	#普通实例方法的self指向调用方法的对象
    def jump(self):
        print("正在执行jump方法")
    der run(self):
        #在实例方法内，调用另一个实例方法时，不能省略调用的对象self
        self.jump()
        print("正在执行run方法")
#创建对象
p=Person()#在括号里可以填实例对象的初始化参数，由于已经有了默认值，故不再
print(p.name,p.age)#john 22

#调用实例方法，第一个参数self时自动绑定的，因此可以只为第二个形参指定一个值
p.say('love autumn_moon')#love autumn_moon

#对实例变量的操作
#修改实例变量
p.name = 'john_Q'
print(p.name,p.age)#john_Q 22
#增加实例变量，一个p的skills实例变量
p.skills=['programming','swimming']
print(p.skills)
#删除实例变量
del p.name
```

<h5>self</h5>

1. 在类体中，定义的方法默认为实例方法，python会**自动绑定**方法的第一个参数为self，该参数它的指向
   - 构造方法\__init__(self,...)，self的指向正在初始化的对象
   - 普通实例方法的self指向调用此方法的对象

2. 通过实例对象，为对象增加实例方法，python不会自动为该方法绑定self，需要开发者**手动绑定**

   ```python
   def info(self):
       print("--对象p的info函数--",self)
   #法一、手动为增加的实例，绑定第一个参数为实例对象p
   p.foo = info(p)
   #法二、MethodType
   from types import MethodType
   p.foo = MethodType(info,p)
   ```

## 6.2 方法

### 6.2.1 类调用实例方法

类调用实例方法时，python不会自动为实例方法的第一个参数绑定实例对象。必须手动为方法的第一个参数传入参数值（实例方法中，如果第一个参数self用作实例对象，那么传入的参数必须为实例对象）

### 6.2.2 实例调用类方法与静态方法

使用@classmethod修饰的方法就是类方法

使用@staticmethod修饰的方式就是静态方法

他们都推荐使用类来调用（使用对象也可以调用，情形和类调用一致），类方法的第一个参数会自动绑定到类本身cls，静态方法则不会自动绑定cls

编程时，很少用到类方法和静态方法。但在特殊的场景（比如使用工厂模式）下，类方法或静态方法也是不错的选择。

```python
class Bird:
    @classmethod
    def fly(cls):
        print('类方法的第一个参数：',cls)
    @staticmethod
    def info(p)
    	print('静态方法的第一个参数',p)
b = Bird()
#类和对象调用类方法，都会为类方法自动绑定cls
Bird.fly()
b.fly()#实质上，依然还是使用类调用
#类和对象调用静态方法，不会绑定cls，故必须手动传入第一个参数
Bird.info('通过类调用静态方法')
b.info('通过对象调用静态方法')#实质上，依然还是使用类调用
```

 ## 6.3 成员变量

### 6.3.1 类变量

在类体内定义的变量——类变量

python推荐使用类来读写类变量。

**对象对类变量只有读操作，没有写操作。**对象读类变量实质还是通过类名在读。对象想通过赋值修改类变量实质上并未修改类变量，而是重新定义了一个与类变量同名的实例变量。

### 6.3.2 实例变量

在类的方法体内，可通过self增加实例变量。

<h5>property()</h5>

如果为python定义**为实例变量定义了getter，setter等访问器**，则可使用property()函数在类体中将他们定义为**属性（相当于实例变量，常用作计算属性，在后面的封装隐藏中会用到）**

<pre>
    obj_field_name = property(fget=None,fset=None,fdel=None,doc=None)
</pre>

property函数的四个参数分别是getter方法，setter方法，del方法和doc说明文档。参数默认值为None，如果不传默认没有该操作。

<h5>@property</h5>

还可以使用@property装饰器修饰方法使之成为属性。

```python
class RectAngle:
    name='rectAngle' #类变量
    def __init__(self,width,height):
        self.width=width#实例变量
        self.height=height
    def setSize(self,size):
        self.width,self.height=size
    def getSize(self):
        return self.width , self.height
    def delSize(self):
        self.width=0
        self.height=0
#计算属性：size
size=property(getSize,setSize,delSize,'用于描述矩形大小的属性')

print(RectAngle.size.__doc__)
help(RectAngle.size)

rect=RectAngle(4,3)
print(rect.size)#(4,3)

rect.size=6,8
print(rect.width)#6
print(rect.height)#8

del rect.size
print(rect.width)#0
print(rect.height)#0


@property#为state属性定义getter方法
def state(self):
    return self._state
@state.setter#为state属性定义setter方法
def state(self,val):
    if 'alive' in value.lower():
        self._state='alive'
    else:
        self._state='dead'
#如果只定义了如上两个方法，那么属性只有读写两个方法
c=Cell()
c.state='alive'
print(c.state)#alive
```

## 6.4 封装

为了实现良好的封装，需要：

1. 将对象的属性和实现细节隐藏起来，不允许外部直接访问
2. 把方法暴露出来，让方法来控制这些属性进行安全的访问和操作

python并没有提供类似于其他语言的private等修饰符，因此**python并不能真正支持隐藏。**

为了隐藏类中的成员，python完了一个小技巧：只要将python类的成员命名为以**双下划线开头**的，python就会把他们隐藏起来。

```python
class User:
    def __hide(self):
        print('示范隐藏hide方法')
    def getName(self):
        return self.__name
    def setName(self,name):
        if len(name)<3 or len(name)>8:
            raise ValueError('用户名长度必须在3~8个字符之间')
       	self.__name=name
	name=property(getName,setName)
u=User()
u.name='fk' #引发value错误：用户名长度必须在3~8之间
u.__hide() #AttributeError:'User' object has no attribute '__hide'

#python其实没有真正的隐藏机制
u._User__hide()#示范隐藏的hide方法
u._User__name='fk'
print(u.name)#fk
    
```

## 6.5 继承

python的继承是多继承机制。推荐尽量使用单继承。

<pre>
    class subClass(superClass1,superClass2,...):
    	类体
</pre>

object类是所有类的父类。如果定义一个类时，未显示指定直接父类，则默认object类。

子类扩展（extend）了父类，父类派生（derive）出子类

### 6.5.1 多继承

当一个子类有多个直接的父类时，子类会继承得到所有父类的方法。

- **如果父类中包含了同名的方法，此时排在前面的父类中的方法会“遮蔽”排在后面的父类中的同名方法（就近原则）**
- **子类也会继承得到父类的构造方法，如果子类有多个直接父类，那么排在前面的父类的构造方法会被优先使用（就近原则）**

### 6.5.2 重写方法

子类包含与父类同名的方法的现象被称为**方法重写（Override）**，也被称为方法覆盖，符合**就近原则**。

**如果在子类中需要调用父类中被重写的方法，可以通过父类来调用实例方法（第一个参数不会自动绑定self，需要手动）——未绑定方法**

### 6.5.3 super()与构造方法

1. 子类**没有重写**构造方法——如果子类有多个直接父类，那么排在前面的父类的构造方法会被优先使用（就近原则）

2. 子类**重写**了构造方法——python要求：如果子类重写了父类的构造方法，那么子类的构造方法必须调用父类的构造方法。

   子类调用父类方法有两种方式：

   - 使用未绑定方法
   - 使用**super()**函数调用父类的构造方法。

super()的本质就是调用super类的构造方法来创建super对象，super对象可以调用父类的实例方法和类方法(包括构造方法），并且能自动绑定第一个参数为self和cls。

<pre>
    super()
    super(type)#type为类名，通过类名可以区分是哪一个类
    super(type,obj)#要求obj是type类的实例。
</pre>


通过调用不同父类的super(type)可以同时初始化多个父类的实例变量，以保证不出错。

## 6.6 多态

对于弱类型的语言来讲，变量并没有声明类型，因此同一个变量完全可以在不同时间引用不同的对象。

**当同一变量在调用同一方法时，完全可以呈现出多种行为（具体呈现出哪种行为由该变量所引用的对象来决定）——多态（Polymophism)**

## 6.7 动态性

python是动态语言，动态语言的典型特征就是：类、对象的属性和方法都可以动态增加和修改。

### 6.7.1 \__slots__和动态属性

**如果希望为所有的实例都添加方法，则可以通过为类添加方法实现。**

动态性固然有好处，但程序定义好的类有可能被后面的程序修改，这就带来了不确定性。

如果程序要限制为某个类动态添加属性和方法，则可以通过\__slots__属性来指定

<h5>__slots__</h5>

1. \__slots__属性的值是一个元组，该元组的所有元素列出了**<u>该类的实例允许动态添加的所有属性名和方法名</u>**。

2. \__slots__属性并不限制通过类来动态添加属性和方法。
3. \__slots__属性只对当前类的实例有效，对它的子类无效。

### 6.7.2 type()和动态类

type()可以查看变量（包括对象）的类型。

实际上python完全允许使用type()函数来创建type对象，又由于type类的实例就是类，因此python可以使用type()来**<u>动态创建类</u>**

<pre>
    type(className , (superClass1 , superClass2 , ...),dict(field1,..fun1,...))
</pre>

### 6.7.3 类型检查

python提供了如下两个函数来检查类型

1. issubclass(cls, class_or_tuple)：检查cls是否为后一个类或元组包含的多个类中任意类的子类。
2. isinstance(obj , class_or_tuple)：检查obj是否为后一个类或元组包含的多个类中任意类的对象。
3. Python为所有类提供了一个\__base__属性，通过该属性可以查看该类的所有直接父类，该属性返回直接父类组成的元组。

## 6.8 枚举类

实例有限且固定的类——枚举类

程序有两种方式定义枚举类：

1. 直接使用enum.Enum()方法列出多个枚举值来创建简单枚举类。
2. 通过集成enum.Enum基类来派生复杂枚举类（带方法）。

枚举值都是该枚举类的成员变量，也是枚举类的实例对象，每个成员对象都有name（该枚举值得变量名），value（枚举值的值，默认为枚举值得序号，通常从1开始）两个属性

```python
#法一
import enum
Season=enum.Enum(Season,('spring','summer','autumn','winter'))#第一个参数，枚举类类名，第二个参数，是一个元组，用于列举所有枚举值
#直接访问枚举对象
print(Season.spring)
#访问枚举成员的变量名
print(Season.spring.name)
#访问枚举成员的值
print(Season.spring.value)
#通过枚举变量名访问枚举对象
print(Season['spring'])
#通过枚举值访问枚举对象
print(Season(1))

#法二
import enum
class Orientation(enum.Enum):
    #为序列值指定value
    East='东'
    South='南'
    West='西'
    North='北'
    def info(self):
        print('这是一个代表%s的枚举' % self.value)
Orientation.East.info()#这是代表东的枚举
```

# 9 模块和包

python3的标准库[参考文档](https://docs.python.org/3/library/index.html)

## 9.1 模块化编程

模块就是python程序，模块文件的文件名就是他的模块名。

对于一个真实的python程序，我们不可能自己完成所有的工作，通常需要借助于第三方类库。此外也不可能在一个源文件中编写整个程序的源代码，这些都需要以模块化的方式来组织项目的源代码。

### 9.1.1 导入模块

import 语句主要有两种用法：

1.导入整个模块(所有成员)

<pre>
    import module1 [as alias1], module2 [as alias2],....,模块名n [as 别名n]
</pre>

2.导入模块中的指定成员

<pre>
    form module import member1 [as alias1], member2 [as alias2],....,成员名n [as 别名n]
</pre>

当使用第一种import 语句导入模块中的成员时，必须添加模块名或模块别名作为前缀

当使用第二种import 语句导入模块中的成员时，直接使用成员名或成员别名即可

```python
import sys as s, os as o
print(s.argv[0],o.sep)#输出程序名，分隔符、
from sys import argv as v
print(v[0])
```

<h5>导入模块的本质</h5>

- 使用 import module 导入模块的本质就是：将module.py中全部的代码加载到内存中并执行，然后将整个模块内容赋值给与模块同名的变量，该变量的类型是module，而在该模块中定义的所有程序单元都相当于该module对象成员。

- 使用from module import member导入模块的本质：将module中的全部代码加载到内存并执行，然后只导入指定成员，并不会将整个模块导入

<h5><pre>模块的__all__变量</pre></h5>

模块的\_\_all\_\_变量，将变量的值设置成为一个列表，只有该列表的程序单元或成员变量才会被暴露到模块之外

```python
def hello():
    print('hello')
def world():
    print('this is python world')
def test():
    print('this test unit')
__all__=['hello','world']
#引入的模块无法from module import * 引入所有成员，使用test会报错
```

使用\_\_all\_\_列表之外的成员，可以通过import module 前缀加成员名调用，也可以通过from module import specialmember 调用程序单元

### 9.1.2 定义模块

模块就是python程序，模块文件的文件名就是他的模块名。

<h5>为模块编写说明文档</h5>

在模块开始处定义一个字符串直接量即可，即在第一行代码之前

```python
'''
模块说明
name：属性名
add:增加
del：减少
'''
import sys as s
......
```

<h5>为模块编写测试代码</h5>

当编写模块完成的时候，需要测试代码是否无误。当模块被其他模块引入时，又无需测试该模块代码。

此时可借助所有模块内置的\_\_name\_\_变量区分，如果直接使用python命令来运行一个模块，\_\_name\_\_变量值为\_\_main__；如果该模块被导入到其他模块中，\_\_name\_\_变量的值就是模块名；

即我们可以在模块中添加一段如下代码

```python
if __name__ =='__main__' :
    #测试代码块
```

### 9.1.3 加载模块

在编写一个python模块之后，如果直接使用import或from...import 来导入模块，python通常并不能加载该模块，道理很简单：python怎么知道去哪里找到这个模块呢？

为了让python找到我们编写（或第三方提供）的模块，可以用以下两种方式来告诉它。

- 使用环境变量
- 将模块放在默认的模块加载路径下。（lib\site-packages）路径，它专门用于存放python的扩展模块和包

## 9.2 包

为了更好地管理多个模块源文件，python提供了包的概念。

什么是包？

- 从物理角度看，包就是一个文件夹，该文件夹下包含了一个\_\_init\_\_.py文件，该文件夹可以包含多个模块源文件
- 从逻辑上来看，包的本质依然是模块。导入包和导入模块的语法完全相同

### 9.2.1 定义包

- 创建一个文件夹，该文件夹的名字就是该包的包名。
- 在该文件夹内添加一个\_\_init\_\_.py文件即可

使用import package 导入包的本质就是加载并执行_\_init\_\_.py文件，然后将整个文件内容赋值给与包同名的变量，该变量的类型是module。

在包内创建多个module.py文件，然后在_\_init\_\_.py编辑

```python
from . import module1
from .module1 import *
```

有相对路径导入方法

# 常用函数

1. **id( obj )**：
   - 查看对象内存地址
2. 