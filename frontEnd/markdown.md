# Markdown

# 1. 标题

    h1~h6 n个'#' 开头 空格隔开，h1一下面有横线，可两头加n个'#',空格隔开，形成闭合标签，按对应的字体书写内容，
    
    内容换行需要回车两下，中间要空一行，不然换行后，内容还是连续的。
    
    内容层级化操作，采用tab键
    
    换样式顶格书写

<b> html的语法标签和规范在md文件依然适用</b>
<!--注释-->
<b>加<br/>粗</b>

# 标题一
## 标题二
### 标题三
#### 标题四
##### 标题五
###### 标题六

# 2. 列表
    分有序和无序列表。
## 2.1 无序列表
    用 +|-|* 开头。

+ 列表一(无序列表'+ -'都可以)
    + xxx
        + xx
        + xx
    - xx

## 2.2 有序列表
    阿拉伯数字+'.'开头，空格隔开。特别注意，有序列表的序号是根据第一行列表的数字顺序来的
1. 列表一
2. 列表二
1. 列表三

# 3.区块引用 

    对某个部分做的内容做一些说明或者引用。
    
    引用因为是一个区块，理论上是应该什么内容都可以放，如标题，列表，还可嵌套使用等等。
    
    多个引用间要做换行操作。

>摘自新鸳鸯蝴蝶梦
>
>>原作：李煜

>昨日像那东流水，你我远去不可留
>
>>多级引用

# 4.分割线
分割线可以由* | - 至少要3个，且不需要连续，有空格也可以

---
- - -
***
* **

# 5.链接
    支持2种链接方式：行内式和参数式
## 5.1 行内式
    [官网链接](www.baidu.com)

[官网链接](http://www.baidu.com)
## 5.2 参数式
[官网链接]:www.baidu.com "百度搜索"

这里是 [官网链接]

# 6.图片
   图片也有2种方式：行内式和参数式

![图片](https://www.cnblogs.com/images/logo_small.gif)

[博客园]: https://www.cnblogs.com/images/logo_small.gif

参数式图片：这里是![博客园]

如果图片不在服务器上，图片的引用也可以使用相对路径。

比如在当前文件路径之下有一个pictures文件夹，文件夹中有你需要的图片girl.png。

你可敲代码`![](./pictrues/girl.png)`。就可以在此md文件中引用相对路径中的图片。

一个点代表当前路径下，两个点代表上一层文件路径（父级文件夹），可重复使用，eg：`../../xx.png`，爷级路径

# 7.代码框

```javascript
console.log('反引号里放代码');      //注释内容
```
```c
int a=100;
long int c=10000;
if(a>100){
    a+=1;
}
```
文字内容也可以写在里面，模板语法。
```
- Dashboard
  - 分析页
  - 监控页
  - 工作台
- 表单页
  - 基础表单页
  - 分步表单页
  - 高级表单页
- 列表页
  - 查询表格
  - 标准列表
  - 卡片列表
  - 搜索列表（项目/应用/文章）
```

# 8.表格
    冒号表示对齐方式，分割线左边左对齐，右边右对齐，两边都有居中对齐


Name | Lunch order | Spicy | Owes
:------- | :----------------: | ----------: | ---------:
Joan  | saag paneer | medium | $11
Sally  | vindaloo        | mild       | $14
Erin   | lamb madras | HOT      | $5

# 9. 文本
*字体倾斜*
_字体倾斜_

**字体加粗**
__字体加粗__

~~删除线~~
# 10.[公式](https://blog.csdn.net/mingzhuo_126/article/details/82722455)

[markdown公式符号大全](https://blog.csdn.net/konglongdanfo1/article/details/85204312)

"$$"+enter
"{}"——为公式块

"\\"——为转义符

"\\\\"——双反斜杠换行


$$
x^2+y_1+\frac{1}{2}+\sqrt{2}+\log_{2}{8}+\vec{a}+\int_{1}^{2}{x}dx+\lim_{n\rightarrow+\infty}{\frac{1}{n}}+\sum_{n=1}^{100}{a_n}+\prod_{n=1}^{100}{x_n} 
\\ \pi+\alpha+\beta+\gamma
\\ x \in A
\\
\frac{\partial f}{\partial w_1} = \frac{\partial f_3}{\partial f_2} w_3 \times \frac{\partial f_2}{\partial f_1} w_2 \times
\frac{\partial f_1}{\partial w_1}
$$
$$
\vec{a}  向量\\
\overline{a} 平均值\\
\widehat{a} (线性回归，直线方程) y尖\\
\widetilde{a} 颚化符号  等价无穷小\\
\dot{a}  一阶导数\\
\ddot{a}  二阶导数
$$



设置对对齐——begin{align}，end{align}，&标记对齐位置
$$
\begin{align}
h_{t-1}：& t-1时刻的隐藏层
\\
x_t：&t时刻的特征向量
\\
h_t：&加softmax即可作为真正的输出，否则作为隐藏层
\end{align}
$$



## [矩阵](https://www.jianshu.com/p/734c742c1331)

$$
\left[
\begin{matrix}
a & b & c & d  \\
e & f & g & h  \\
i & j & k & l  \\
m & n & o & p
\end{matrix}
\right]
\tag{2}
$$


$$
\left(
\begin{matrix}
a & b & c & d  \\
e & f & g & h  \\
i & j & k & l  \\
m & n & o & p
\end{matrix}
\right)
\tag{3.1}
$$

$$
A=\begin{pmatrix}
a & b & \cdots & c  \\
d & e & \cdots & f  \\
\vdots & \vdots & \ddots & \vdots  \\
g & h & \cdots & j
\end{pmatrix}
\tag{5.1}
$$




# 11.锚点

```txt
跳转
方法1 ： [跳到标题一](#123)
方法2 ： <a href='#123'>跳到标题一</a>

锚点：<a name='123'>我是标题一</a>
```

[跳到标题一](#123)

<a href='#123'>跳到标题一</a>

h

t

m

l

m

m

m

m

m

hh

j

h

h

h

h

h

h

h

h

h

<a name='123'>我是标题一</a>

ff

f

f

f

f

# 其他注意

1. enter——是双换行
2. 【shift+enter】——是单换行
3. 【ctrl + 加号或减号】：用于放大字号或缩小字号

# 参考

1. https://www.jianshu.com/p/335db5716248
2. https://blog.csdn.net/mingzhuo_126/article/details/82722455



# [Mermaid](https://www.jianshu.com/p/6e46a1498e4c)

Mermaid 是一种类似于 Markdown 的语法，您可以在其中使用文本来描述和自动生成图表。使用 Mermaid 的受 Markdown 启发的语法，您可以生成流程图、UML 图、饼图、甘特图等。









