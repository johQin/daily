# XML-可扩展标记语言

XML旨在传输、存储数据（小型数据库）和配置文件，作为HTML（旨在显示数据）的扩展，xml标签没有预定义，需要自行定义标签，语法严格。

xml文件的文档声明必须在第一行，用于指定版本和字符集，如下写在第一行。

```xml
<?xml verson="1.0" encoding="UTF-8"?>
```

# 1 XML基本语法

1. 所有标签都必须有关闭标签，标签对大小写敏感，必须有根元素，属性值必须用引号括

2. 实体引用：一些字符有特殊的意义，我们不得在其他意义上使用它们，想要引用它们的实体，必须通过实体引用，

   五个预定义的实体引用：

   - \&lt;小于符<
   - \&gt;大于符>
   - \&amp;&符
   - \&apos;单引号
   - \&quot;双引号

3. 元素：命名规则和变量命名规则相同，并且不能以xml开头	

4. 文本：文本原样显示,一些和标签类似的文本无法原样显示，这里就需要原样显示标签。这个标签会阻止浏览器解析，而直接显示文本。

    eg:\<![CDATA[需要原样显示的内容]]>

# 2 XML约束

规定标签文档内只能写哪些标签，并且给一些提示。

1. 约束分类：DTD（Document Type Definition 文档类型定义）和schema（纲要）	

2. DTD约束：主要是在.xml文件外建立.dtd文件，然后引入.xml文件
  - 引入格式：

       ```xml
       <!DOCTYPE 被约束的标签名 SYSTEM "约束文件.dtd">
       ```

  - 约束语法：
       eg:	

       ```xml
       <!ELEMENT web-app(servlet*,servlet-mapping*,welcome-file-list?)>
       <!ELEMENT servlet(servlet-name,description?,(servlet-class|jsp-file))>	
       <!ELEMENT servlet-name(#PCDATA)>//约束他的内容必须是文本
       <!ELEMENT welcome-file-list(welcome-file+)>
       ```

       ps：子标签必须按顺序出现
           		()外根标签，()内子标签，
           		*可用任意次，?可以出现并且只有一次，无符号必须出现并且只有一次
           		|两侧元素选一必须出现，+至少有一个标签

       ```xml
       <!ATTLIST student number ID #REQUIRED>//添加属性			
       ```

  - 局限性：无法规定标签中内容的类型

3. schema(.xsd文件)

   能规定标签内容的类型，可以规定的很详细。具体的使用请参阅文档

   - 引入格式： 

     - (1)编写根标签

     - (2)引用约束地址

     - (3)引用实例名称空间

     - (4)默认名称空间

       eg：\<student 234 步都在此填写>标签内容\</student>

   - 参阅文档

# 3 DOM解析

Document Object Module 文档对象模型

1. xml的文档结构：树状结构，DOM操作的文档节点(.xml)，元素节点(/* </> */)，属性节点(eg:ID)，文本节点(元素内容)
2. DOM解析的特点：
      - a.在加载的时候，一次性的把整个文档载入内存，在内存中形成一颗树（document对象）
      - b.代码中操作的Document对象，其实操作的是内存中的DOM树，和本地磁盘中的xml文件没有直接关系。
      - c.由于操作的是内存中的dom，磁盘中xml的内容并没有改变。当内容数据发生改变时，需要进行同步操作。
      - d.缺点:若xml文件过大，可能造成内存溢出
3. DOM解析步骤：创建解析器工厂=>造解析器=>获取document对象=>获取节点对象=>节点对象.内容//查阅
4. 数据增删改
5. document对象到xml文件的同步操作

## Dom4j解析

是Dom4j是dom4j.org出品的开源xml解析包

dom4j是十分优秀的javaXML API,性能优异、极易使用、需要在网上下载他的java包，推荐使用

## Sax解析

 特点：逐行读取，事件驱动
 优点：不占内存，速度快
 缺点：只能读取，不能回写
 很少用

