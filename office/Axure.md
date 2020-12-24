# Axure RP

Axure RP 是一个专业的快速原型设计工具，Axure是一个公司，RP则是Rapid Prototyping 快速原型的缩写。

原型分类：草图，低保真，高保真

原型设计流程：

1. 需求分析：用户调研，竞品分析，分析用户的反馈和产品数据
2. 页面架构的设计：思维导图理清逻辑关系，流程图表达主要流程任务，产品信息架构设计，页面布局设计
3. 低保真原型设计
4. 原型评审：各个领导针对产品的需求会对产品原型进行评审
5. 高保真原型设计：可以用于给领导汇报或者概念性产品，把低保真原型，经过视觉设计师制图、切图，然后制作出高保真原型

# 1 页面管理

作用：

1. 规划软件的功能单元或者软件结构
2. 让软件使用者快速了解软件结构，快速定位页面

## 1.1 管理概览

页面区域注意：

1. 制作软件的原型时，要规划软件的功能菜单或者栏目结构
2. 页面的命名要有意义

![](./legend/RP/页面管理.png)

## 1.2 蓝月亮官网结构设计

![](./legend/RP/蓝月亮官网栏目结构设计.png)

# 2 元件

## 2.1 线框图元件

元件分类：

1. 基本型元件
2. 表单类元件
3. 菜单表格类元件
4. 标记元件

**各个元件右击都会有相应的特殊功能**

### 2.1.1 基本型

![](E:/gitRepository/daily/office/legend/RP/%E5%9F%BA%E6%9C%AC%E7%B1%BB%E5%85%83%E4%BB%B6.png)



- box，Ellipse右击可以变换形状
- image，可以裁剪crop，分割slice
- **hot spot热区**：用于对一个整体的组件其上分割成不同的区域，对其上覆盖的不同区域做不同的事件处理

### 2.1.2 表单型

![](./legend/RP/表单型元件.png)

- Text Field文本框：在右侧的interactions，还可以设置输入类型，hint提示语，交互事件等
- Radio Button单选按钮：选中多个单选按钮，然后设置单选按钮组，可以实现只可选中一个

### 2.1.3 菜单表格型

![](./legend/RP/菜单与表格元件.png)

### 2.1.4 标记型

![](./legend/RP/标记元件.png)

- 页面快照：嵌入另一个页面的内容，有点类似于iframe

## 2.2 流程图元件

![](./legend/RP/流程图元件.png)

样式工具：view -> toolbars -> styles toolbar

**箭头的样式设置**就在样式工具栏里面，单双箭头可以通过点击相应的样式即可设置，当需要减少箭头的时候，需要点击箭头设置的第一个图样即可减少箭头

![](./legend/RP/连接线箭头设置.png)



## 2.3 图标元件

![](./legend/RP/图标元件.png)

## 2.4 元件库

### 2.4.1 载入元件库

1. 手动导入元件库***.rplib**

   ![](./legend/RP/手动导入元件库.png)

2. 将元件库文件放置在安装文件夹下（**axure\DefaultSettings\Libraries**），系统会自动读入组件库

   ![](./legend/RP/元件库载入到内置元件库的位置.png)

   

### 2.4.2 自定义元件库

在导入图片集旁边的更多options里面，Edit选项就可以自定义元件库



## 2.5 Dynamic Panel

动态面板主要用于做一些动态效果

1. 显示隐藏（弹出框，对话框的显示隐藏）
2. 滑动效果
3. 拖动效果
4. 状态切换

右击功能项：

1. fit to content：自适应内容
2. scrollbars：滚动条
3. pin to browser：固定到浏览器，在浏览器固定位置展示、
4. 100%width(broswer only)：用于自适应浏览器宽度
5. break away first state：从首个状态脱离，将状态脱离动态面板形成一个独立的元件
6. create master：转换为模板

### 登录模块设计实例

**1.需要实现的效果**

![](E:/gitRepository/daily/office/legend/RP/%E7%99%BB%E5%BD%95%E6%A8%A1%E5%9D%97%E6%95%88%E6%9E%9C.png)

**2.实施**

![]()2.5 个人简历

单选组：比如下面图中的男和女，要按住ctrl，挨个选中，然后右击，**assign radio group**，即可实现一个组只能选一个的效果

![](./legend/RP/个人简历制作.png)

## 2.6 Repeater

中继器元件是用来显示重复的文本，图片、链接。**类似列表渲染。**

从英文名上面可以看出来，它是用来显示重复的，有规律可循的文本、图表，并且动态管理。

它可以用来模拟数据库操作，进行数据库的增删改查操作

经常会使用中继器来显示商品列表信息、联系人信息、用户信息等。

![](./legend/RP/中继器.png)

### 员工管理实例

#### 绑定数据

![](./legend/RP/中继器绑定数据.png)

#### 添加弹出框设计

增加一个动态面板，设置动态面板的尺寸（x：0，y：0，w：1200，h：800），在动态面板里，添加遮罩层（长宽大小都和动态面板的大小一致，半透明），然后通过box画框，拖拽form组件并命名，关闭标签（set hide隐藏动态面板，bring to back置于底层动态面板，form元件set text为空，）

![](./legend/RP/中继器新增弹框设计.png)

#### 添加人员

![](./legend/RP/中继器添加行.png)

#### 删除人员

![](./legend/RP/中继器删除.png)

# 3 变量

命名只能包含字母和数字，并且以字母开头。

## 3.1 全局和局部变量

### 全局变量

在交互的启用情形中，叫做value of variable

设置：project -> global variable

onLoadVariable是系统的全局变量，作为项目中最少有一个全局变量使用。

![](./legend/RP/全局变量.png)

### 局部变量

局部变量**属于元件widget本身**，通常在交互中作为计算属性使用

![](./legend/RP/局部变量.png)

### 变量在页面间传递

需要达到的效果：

![](./legend/RP/页面间传递数据.png)

登录页设置全局和局部数据

![](./legend/RP/登录页设置数据.png)

登录页按登录跳转到首页，传递数据给矩形框

![](./legend/RP/登录传递给首页的数据.png)

## 3.2 简易计算器

enable case 启用情形

![](./legend/RP/事件触发情形.png)

# 4 母版

## 4.1 创建母版

master，像导航菜单，版权信息等部分为多个页面共有，就可以建立母版使用。

1. 通过在母版区新建

   ![](./legend/RP/母版区新建母版.png)

2. 通过元件右击转换为模板（create master）

## 4.2 drop behavior

拖放行为

在应用母版到页面时，里面有一个**“固定到模板的位置（lock to location in master ）”**，应用到页面后，页面中的母版部分按在原模板的位置放置并无法拖动，除非右击去掉**lock to master location**的勾选。

在页面中右击母版部分->脱离母版（break away），即可与母版断开关系，母版再有任何改变，都不会同步到页面上

## 4.3 实例

**效果图**

![](./legend/RP/蓝月亮导航效果图.png)

**效果实现**

![](./legend/RP/蓝月亮导航的实现.png)

# 5 动作action

针对于浏览器事件所做出的的反应和处理，相当于事件处理函数，在这里不再是写代码，而是通过interaction交互来实现交互

## 5.1 link actions

### 5.1.1 open link

打开链接

![](./legend/RP/链接打开的方式.png)



### 5.1.2 open link in frame

内联框架

类似于iframe标签的元件inline frame

![](./legend/RP/内联框架打开页面.png)



### 5.1.3 scroll to widget

滚动到元件（anchor link），页面常常有锚点可以用这个

![](./legend/RP/锚点anchor.png)

### 5.1.4 自适应视图

**project->adaptive views sets**

## 5.2 widget actions

### 5.2.1 show/hide

1. 切换方式控制显示与隐藏（widget actions -> show/hide toggle visibility）
2. 变量方式控制显示与隐藏（启用情形，加other actions->set variable value）

通过变量的方式可以做很多事情，比如说多级菜单的联动。

![](./legend/RP/导航变量控制.png)

### 5.2.2 set text/image

![](./legend/RP/settext_image.png)

### 5.2.3 set selected/checked

对单选框和复选框进行事件动作的选中

![](./legend/RP/setselected_checked.png)

### 5.2.4 set selected list option

![](./legend/RP/setlistoption.png)

### 5.2.5 enable/disable

文本框，单选按钮，复选框，下拉列表都可以设为启用禁用

![](./legend/RP/启用禁用.png)

### 5.2.6 move/rotate/bring to front/back

移动/旋转/置于顶层/底层

![](./legend/RP/移动旋转层序.png)

### 5.2.7 focus/expand collapse tree node

获得焦点，展开折叠树节点

![](./legend/RP/焦点树节点.png)

# 其他

## 交互interactions面板

在view视图->panes面板->interactions交互可以打开，也可以toggle right panes（将会打开styles，interactions，notes）

![](./legend/RP/interactions.png)

## 格式刷

CTRL + M

![](./legend/RP/格式刷.png)

## 词汇

1. interactions：交互，
2. hint：提示
3. toggle：切换，转换
4. style：样式
5. note：说明

## 惯用手法

1. 边框不显示：线宽设置为0