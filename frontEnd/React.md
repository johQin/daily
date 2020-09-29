# [React基础](https://react.docschina.org/docs/getting-started.html)
<p style='text-indent:2em;'>
学习前端框架，脑中一定要有三个概念，一组件，二生命周期，三数据驱动。前端样式和逻辑处理是由一个个组件组成的。每个组件都有它的生命周期（存在的时间）。页面的样式改变本质是它背后的数据改变引起的，页面由数据驱动的。
</p>

# 1 组件
---
<p style='text-indent:2em;'>
组件是React的核心概念，是React应用程序的基石。组件将应用的UI拆分成独立的、可复用的模块，React应用程序正是由一个一个组件搭建而成的。
</p> 
<p style='text-indent:2em;'>
定义一个组件有两种方式，使用<span style='color:red'>ES6 class（类组件，动态组件）和使用函数（函数组件，静态组件）</span>。</p>

使用class定义组件需要满足两个条件：
1. class继承自React.Component。
2. class内部必须定义render方法，render方法返回代表该组件UI的React元素。

```js
import React from 'react';
class ClassComponent extends React.Component{
    //每个React组件当使用到props的时候，都必须要有constructor函数，而且必须要有以下要素。
    //后面会在生命周期说到
    constructor(props){
        super(props)
    }
    //render里必须包含return并返回一个根React元素。
    //根React元素不换行，不需要加括号。
    //换行用括号，将根节点包住。
    render(){
        return(<div>我是根React元素</div>)//只能有一个根节点
    }
}
```

使用函数组件：
1. 依照函数的形式定义。
2. 函数的返回值是一个React元素

```js
const FunComponent=(a)=>{
    a=a*100;//函数组件作为静态组件，只是根据传入值，计算后显示出来。
    //这里涉及到JSX语法。后面作为会有讲到。
    return(<div>{`计算结果为${a}`}<div>)
}
```

<p style='text-indent:2em;'>
React组件与React元素区别：React元素是一个JS对象，React组件是由多个函数组成，它是UI描述和UI数据的完整体，它会返回一个根React元素，一个根React元素是由若干React元素组建而成的，供页面展示。
</p>

# 2 JSX语法
---
<p style='text-indent:2em;'>
React致力于通过组件的概念将页面进行拆分并实现组件复用。React 认为，一个组件应该是具备UI描述和UI数据的完整体，不应该将它们分开处理，于是发明了JSX，作为UI描述和UI数据之间桥梁。这样，在组件内部可以使用类似HTML的标签描述组件的UI，让UI结构直观清晰，同时因为JSX本质上仍然是JavaScript，所以可以使用更多的JS语法，构建更加复杂的UI结构。
</p>

## 2.1 基本语法
<p style='text-indent:2em;'>


&emsp;&emsp;JSX的基本语法和XML语法相同，都是使用成对的标签构成一个树状结构的数据。


```js
const element = (
            <div>
                <h1>Hello, world!</h1>
            </div>
        )
```
</p>

## 2.2.标签类型
<p style='text-indent:2em;'>

&emsp;&emsp;在JSX语法中，使用的标签类型有两种：<span style='color:red'>DOM类型的标签（div、span等 <span style='color:green'>使用时，标签的首字母必须小写</span>）和React组件类型的标签（<span style='color:green'>使用时，组件名称的首字母必须大写。</span>）</span>。React 正是通过首字母的大小写判断渲染的是一个DOM类型的标签还是一个React组件类型的标签。这两种标签可以相互嵌套。
```js
//React组件
//函数（静态）组件
//标签形式的写法
const HelloWorld=(props)=><h1>Hello,World....{props.a}</h1>
//调用时,
//<HelloWlord a={12}/>
//函数式写法
const helloWord=(props)=><h1>Hello,World....{props}</h1>
//调用时,
//<div>{helloWorld(12)}</div>

//常量组件
const helloWord=(<div>Hello,World</div>)
//调用时
//<div>{helloWorld}</div>

//函数式组件和常量组件在调用时，都是在UI描述里嵌入JavaScript语句，后面会讲到

//类组件
class HelloWorld extends React.Component{
    constructor(props){
        super(props);
    }
    render(){
        return(
            <h1>Hello,World<h1>
        )
    }
}

```
</p>

## 2.3 JavaScript表达式
<p style='text-indent:2em;'>
JSX可以使用JavaScript表达式，因为JSX本质上仍然是JavaScript。在JSX（UI描述）中使用JavaScript表达式需要将表达式用大括号“{}”包起来。<span style='color:red'>而且只能使用表达式（单行JavaScript语句），而不能使用多行JavaScript语句</span>

```jsx
//给标签属性赋值
//给style赋一个对象
const element=<h1 style={{color:red}}>一级标题</h1>
//计算后显示2
const element=<div>{1+1}</div> 
```
</p>

## 2.4 常用渲染方式
<p style='text-indent:2em;'>
在前端中，我们通常把组件显示在页面的过程叫做<b style='color:red'>渲染过程</b>(render函数就是用来渲染页面的)。当然这个过程会在UI描述中使用JavaScript语句，要遵循JSX语法。

&emsp;&emsp;在实际的应用场景中，我们经常会用到两种渲染方式，一、条件渲染，二、列表渲染

### 2.4.1 条件渲染

    当条件成立时，我们就在页面上渲染组件，通常有以下三种写法：
```jsx
class HelloWorld extends React.Component{
    constructor(props){
        super(props);
        this.state={
            flag:true,
            flag0:2,
        }
    }
    render(){
        
        return(
        <div>
        {/*条件渲染，可嵌套使用*/}
            <div>
            { flag&&<div>条件成立时显示</div> }
            </div>
            <div>
            { flag?<div>条件成立时显示</div>:<div>条件不成立时显示</div>}
            </div> 
            <div>
            { flag?<div>条件成立时显示</div>:
            (flag0==2?<div>条件不成立时显示且flag0=2时显示</div>:<div>条件不成立时显示且flag0!=2时显示</div>))}
            </div>
        </div>)
    }
}
```

### 2.4.1 列表渲染
<p style='text-indent:2em;'>
    当页面需要渲染如下面的列表的时候，从图中可以看到列表里的每条记录的样式差不多一致。列表在数据上面体现的是数组。我们可以通过代码的循环结构，遍历数组的每一项，然后渲染在页面上。
</p>

![图片](./listRender.png)

这里我们会使用到js的[map()](https://www.runoob.com/jsref/jsref-map.html)函数。

```jsx
class HelloWorld extends React.Component{
    constructor(props){
        super(props);
        this.state={
            arr:[
                '1、任天堂新款Switch',
                '2、大疆通过美国审核',
                '3、杭州失联女童监控画面',
                '4、荣耀将进军电视行业',
                '5、高校替课月入数千',
            ]
        }
    }
    render(){
        const {arr}=this.state
        return(<div >
        {
            arr.map((item,index)=>{
            let color='blue;
            switch(index){
                case 1:
                color='red'; 
                case 2:
                color='yellow';
                case 3:
                color='orange';
            }
            return <div style={{color}} key={index}>{item}</div>
            // 假如使用列表渲染，则循环的根节点，必须有唯一的key属性
        })}
        </div>)
    }
}
```

</p>

# 3 生命周期
---
<p style='text-indent:2em;'>
<b>组件从被创建到被销毁的过程称为组件的生命周期</b>。React为组件在不同的生命周期阶段提供不同的生命周期方法，让开发者可以在组件的生命周期过程中更好地控制组件的行为。需要提醒的是只有React class组件才有生命周期函数。通常，组件的生命周期可以被分为三个阶段：<span style='color:red'>挂载阶段、更新阶段、卸载阶段。</span>
</p>

![生命周期图解](./reactLifeCircle.png)

## 3.1. 挂载阶段

<p style='text-indent:2em;'>
这个阶段组件被创建，执行初始化，并被挂载到DOM中，完成组件的第一次渲染。依次调用的生命周期方法有：
</p>

### 3.1.1 constructor

    如果不初始化 state 或不进行方法绑定，则不需要为 React 组件实现构造函数。
```js
constructor(props){
    super(props);
    this.state={
        flag:true,
        visible:false,
        role:1,
    }
    
    //普通属性
    //当我们组件需要用到一个变量，并且它与组件的动态渲染无关时，就应该把这个变量定义为组件的普通属性。
    this.columns=[
        {title:'姓名',key:'name'},
        {title:'年龄',key:'age'},
    ]

    //事件的处理函数或者其他方法，如果用箭头函数定义，则可以省去这个绑定过程。
    this.handleOK = this.handleOK.bind(this);
   
}
render(){
    //state的调用方式，props和state相似,都是只读的，不可给赋值。
    //这样写是错误的
    //this.props.location='你赋值的样子，不是很像爱情';

    const {visible} =this.state;
    const role=this.state.role;
    return(
        <div>{this.state.flag}</div>
        <h1>{visible}</h1>
        <h1>{role}</h1>
    )
}
```
<span style='color:red;font-size:20px'>props和state</span>


1. 组件的props

<p style='text-indent:2em;'>
<span  style='color:red;'>组件对外的数据接口</span>，通常用来接收组件外传入的数据（包括方法），例如父组件传来的数据，以及后台传来的数据。里面包含一些js基本的location，match，history属性。如果采用其他的框架，可能还有其他封装的属性在里面。例如antd pro的项目框架，后面会着重说到。你必须在这个方法中首先调用super(props)才能保证props被传入组件中。<span  style='color:red;'>props是只读的，你不能在组件内部修改props</span>
</p>

2. 组件的state
<p style='text-indent:2em;'>
<span  style='color:red;'>组件对内的数据接口</span>，state是可变的，组件状态的变化通过修改state来实现。state的变化最终将反映到组件UI的变化上。我们在组件的构造方法constructor中通过<span  style='color:red;'>this.state定义组件的初始状态，并通过调用this.setState（异步的，不要依赖当前state计算下一个state）方法改变组件状态（也是改变组件状态的唯一方式）</span>，this.state进而组件UI也会随之重新渲染。
通过this.state.xxx=val;改变state中的值，但它不会引起组件的重新渲染。
</p>
<p style='text-indent:2em;'>
<b>动态组件和静态组件</b>，这里说的态就是state，并不是所有组件都需要state，当一个组件的内部样式不发生变化的时候，就无需使用state。函数组件就是静态组件，而动态组件就通常就是React class组件。
</p>
<br/>

<span style='color:red;font-size:20px'>this.setState(updater, callback)</span>
<div style='text-indent:2em;'>
通知 React 需要使用更新后的 state 重新渲染此组件及其子组件。这是用于更新用户界面以响应事件处理器和处理服务器数据的主要方式。

updater——可以是一个返回对象的函数也可以直接就是一个对象
```js
function(state,props){ 
    ...;
    return {
        state1:val1 
    }
}
```
callback——组件渲染更新后被执行的回调函数
</div>

### 3.1.2 componentWillMount

<p style='text-indent:2em;'>
这个方法在<span  style='color:red;'>组件被挂载到DOM前调用</span>，且只会被调用一次。这个方法在实际项目中很少会用到，因为可以在该方法中执行的工作都可以提前到constructor中。在这个方法中调用this.setState不会引起组件的重新渲染。
</p>

### 3.1.3 render

<p style='text-indent:2em;'>
这是定义组件时<span  style='color:red;'>唯一必要的方法</span>（组件的其他生命周期方法都可以省略）。在这个方法中，根据组件的props和state返回一个React元素，用于描述组件的UI，通常React元素使用JSX语法定义。需要注意的是，render并不负责组件的实际渲染工作，它只是返回一个UI的描述，真正的渲染出页面DOM的工作由React自身负责。render是一个纯函数，在这
个方法中不能执行任何有副作用的操作，所以不能在<span  style='color:red;'>render中调用this.setState</span>，这会改变组件的状态。
</p>
<span style='color:red;font-size:20px'>组件样式</span>

render作为UI描述的接口，就有必要知道如何调整页面样式。为组件添加样式的方法主要有两种：<span style='color:red;'>外部CSS样式表和内联样式</span>。

1.外部CSS样式表
<p style='text-indent:2em;'>
这种方式和我们平时开发Web应用时使用外部CSS文件相同，CSS样式表中根据HTML标签类型、ID、class等选择器定义元素的样式。唯一的区别是，React元素要使用<span style='color:red;'>className来代替class作为选择器</span>。
</p>

```js
//css Modules，可以解决class名冲突。在antd pro项目中经常会用。
//但在create-react-app创建的项目中，默认配置式不支持这一特性的。
//Template.js
import styles from './Template.css';
const Welcome=<div className={styles.welcome}>hello，Dear ladys</div>
//Template.css
.Welcome{
    width:100%;
    color:red;
    font-size:20px;
}

```

2. 内联样式
<p style='text-indent:2em;'>
内联样式实际上是一种<b>CSS in JS</b>的写法：将CSS样式写到JS文件中，<span style='color:red;'>用JS对象表示CSS样式，对象的样式属性名使用驼峰写法</span>，然后通过DOM类型节点的style属性引用相应样式对象。多用于，样式的变化有赖于组件的状态时。
</p>

```js
const Welcome=<div style={{width:'100%',color:'red',fontSize:20,}}>hello,Dear ladys</div>
//本质上，就是给style赋样式对象
const greeting={width:'100%',color:'red',fontSize:20,}
const Greet=<div style={greeting}>how its' going ? my Sweet</div>
```

<span style='color:red;font-size:20px'>事件处理</span>

在UI描述中，除了展现给用户看的部分，还有对用户的操作做出反应的部分，例如在业务场景中，最常用到的就是用户的点击事件。当然还有很多其他事件，具体可以参看JavaScript的事件处理部分的内容。触发事件操作后，我们需要对此事件进行相应的操作，所以就有了事件处理。

在React元素中绑定事件有两点需要注意：
1. 在React中，事件的命名采用驼峰命名方式，而不是DOM元素中的小写字母命名方式。例如，onclick要写成onClick，onchange要写成onChange等。
2. 处理事件的响应函数要以对象的形式赋值给事件属性，而不是DOM中的字符串形式。


在React事件中，必须显式地调用事件对象的preventDefault方法来阻止事件的默认行为。
```js
 class Baby extends React.Component{
    constructor(props){
       super(props);
       this.state={
           flag:0,
       }
    }
    fresh=(e)=>{
        console.log(e);
        location.reload();
    }
    display=(flag,e)=>{
        console.log(e);
        this.setState({
            flag
        })
    }
    render(){
        const {flag}=this.state;
        return(
            <div>
                <div onClick={()=>{ console.log('此时是诗句',flag)}}>打印当前flag</div>
                <button onClick={this.fresh}>刷新页面</button>
                {()=>{
                    switch(flag){
                        case 1:
                        return <h1>人面不知何处去，桃花依旧笑春风。</h1>
                        case 2:
                        return <h1>无穷无尽是离愁，天涯地角寻思遍。</h1>
                        case 3:
                        return <h1>彼此当年少，莫负好时光</h1>
                        default:
                        return <h1>诗三句欣赏</h1>
                    }
                }}
                
                <button onClick={this.display.bind(this,1)}>显示诗句一</button>
                <button onClick={this.display.bind(this,2)}>显示诗句二</button>
                <button onClick={this.display.bind(this,3)}>显示诗句三</button>

            </div>
        )
    }
}
```


### 3.1.4 componentDidMount
<p style='text-indent:2em;'>
在组件被挂载到DOM后调用，且只会被调用一次。<span style='color:red;'>这时候已经可以获取到DOM结构，因此依赖DOM节点的操作可以放到这个方法中。这个方法通常还会用于向服务器端请求数据。在这个方法中调用this.setState会引起组件的重新渲染。这个方法是比较适合添加订阅的地方。如果添加了订阅，请不要忘记在 componentWillUnmount() 里取消订阅</span>
</p>

## 3.2 更新阶段
 组件被挂载到DOM后，组件的props或state可以引起组件更新。

 &emsp;&emsp;<span style='color:red;'>props引起的组件更新</span>，本质上是由渲染该组件的父组件引起的，也就是当父组件的render方法被调用时，组件会发生更新过程，这个时候，组件props的值可能发生改变，也可能没有改变，因为父组件可以使用相同的对象或值为组件的props赋值。但是，无论props是否改变，父组件render方法每一次调用，都会导致组件更新。

 &emsp;&emsp;<span style='color:red;'>State引起的组件更新</span>，是通过调用this.setState修改组件state来触发的。

 组件更新阶段，依次调用的生命周期方法有：
### 3.2.1 componentWillReceiveProps(nextProps)

<p style='text-indent:2em;'>
<span style='color:red;'>只在props引起的组件更新过程中，才会被调用</span>。nextProps是父组件传递给当前组件的新的props值，nextProps和当前的this.props的值可能相同。setState不会引起此函数发生调用，因此，在此函数中可以调用this.setState，并且不会发生副作用（死循环），此函数引起的state变化，只能在render及其之后的生命周期在this.state看到变化。
</p>

### 3.2.2 shouldComponentUpdate(nextProps,nextState)

<p style='text-indent:2em;'>
这个方法决定组件是否继续执行更新过程。当方法返回true时（true也是这个方法的默认返回值），组件会继续更新过程；当方法返回false时，组件的更新过程停止，后续的componentWillUpdate、render、componentDidUpdate也不会再被调用。一般通过比较nextProps、nextState和组件当前的props、state决定这个方法的返回结果。这个方法可以用来减少组件不必要的渲染，从而优化组件的性能。
</p>

### 3.2.3 componentWillUpdate(nextProps,nextState)

<p style='text-indent:2em;'>
在组件render调用前执行，可以作为组件更新发生前执行某些地方的操作，一般很少用到。
</p>

### 3.2.4 render

<p style='text-indent:2em;'>
执行更新阶段的组件渲染操作
</p>

### 3.2.5 componentDidUpdate(prevprops,prevState)

<p style='text-indent:2em;'>
执行组件更新后的相关操作。这里有两个参数分别代表了，组件更新前的props和state。
</p>

## 3.3 卸载阶段
    组件在被卸载前调用

componentWillUnmount
<p style='text-indent:2em;'>
通常用来清除组件中使用的定时器，恢复数据仓库中的初始数据参数。
</p>

# 4 数据交互
## 4.1 父子组件通信

<p style='text-indent:2em;'>
<span style='color:red;'>
核心：
</span>

- 父传子，通过属性props传递

- 子传父，子通过使用父传递给子的函数，以参数的形式传递给父

</p>

```js
export default class Dad extends React.Component{
    constructor(props){
       super(props);
       this.state={
            a:1,
        }
    }
    getBabyRes=(change)=>{
        const a=this.state.a+change;
        this.setState({
            a
        })
    }
    render(){
        const params={
            a:this.state.a
            toDad:this.getBabyRes,
            command:'儿子，起床'
        }
        return(
            <div>
            <h1>{this.state.a}</h1>
            <Baby 
                ultimatum='再不起床打人了'
                {...params}
            />
            </div>
        )

    }
};


```
```js
 class Baby extends React.Component{
    constructor(props){
        //可以在这里看看props里有什么。
        console.log(props);
        //{ultimatum: "再不起床打人了", toDad: ƒ, command: "儿子，起床"}

        //假如想把传过来的参数放在state里，这里就可以直接赋值，
       super(props);
    }
    componentDidMount(){
        
    }
    toDadInfo=(flag)=>{
        const {getBabyRes}=this.props
        if(falg){
            getBabyRes(1);
        }else{
            getBabyRes(-1);
        }
    }
    render(){
        const {ultimatum,command}=this.props;
        return(
            <div>
                <h1>{command}</h1>
                <h2>{ultimatum}</h2>
                <button onClick={this.toDadInfo.bind(this,true)}>a+1</button>
                <button onClick={this.toDadInfo.bind(this,false)}>a-1</button>
            </div>
        )
    }
}
```
## 4.2 兄弟组件通信
    当两个组件不是父子关系但有相同的父组件时，称为兄弟组件。

<p style='text-indent:2em;line-height:2;'>
兄弟组件不能直接相互传送数据，需要通过状态提升的方式实现兄弟组件的通信，即把组件之间需要共享的状态保存到距离它们最近的共同父组件内，任意一个兄弟组件都可以通过父组件传递的回调函数来修改共享状态，父组件中共享状态的变化也会通过props向下传递给所有兄弟组件，从而完成兄弟组件之间的通信。
</p>
<p style='text-indent:2em;'>
这里只做了解，到时候用到antd pro 框架，兄弟组件之间的通信会很简单。
</p>

## 4.3 [context](https://zh-hans.reactjs.org/docs/context.html)上下文通信

    当组件所处层级太深时，往往需要经过很多层的props传递才能将。所需的数据或者回调函数传递给使用组件。这时，以props作为桥梁的组件通信方式便会显得很烦琐。React提供context的api，可以参看
```js
import PropTypes from 'prop-types';
Dad.childContextTypes={
        add:PropTypes.func,
        a:PropsTypes.number
    }
class Dad extends React.Component{
    //...
    getChildContext(){
        return {
            add:this.handleAdd,
            a:this.state.a,
        }
    }
    handleAdd=()=>{
        const a=this.state.a+1;
        this.setState({
            a
        })
    }
    
    render(){
        return(
            <div>
            <h1>{this.state.a}</h1>
            <Baby />
            </div>
        )
    }
}




Baby.contextTypes = {
    add: PropTypes.func
};
class Baby extends React.Component{
    constructor(props){
        super(props)
    }
    onAdd=()=>{
        this.context.add();
    }
    render(){
        return(
            <button onClick={this.onAdd}>add+1</button>
        )
    }
}

```
    这部分了解即可，antd pro 项目框架中深层次组件传值将有更简单的做法。

## 4.4 服务器通信
    通常建议在componentDidMount去请求后台数据，而在componentWillReceive去更新后台数据。

与后台接口打交道，通常使用<span style='color:red;'>fetch，ajax（XMLHttpRequest），axios</span>，就antd pro项目框架来说，它有封装好的请求方法，并且更加简单实用。这里只做了解。
## 4.5 [ref](https://www.jianshu.com/p/56ace3e7f565)

React提供的这个ref属性，表示为对组件真正实例的引用。

ref可以挂载到组件上也可以是dom元素上,

挂到组件(class声明的组件)上的ref表示对组件实例的引用。不能在函数式组件上使用 ref 属性，因为它们没有实例，挂载到dom元素上时表示具体的dom元素节点。

<span style='font-size:20px;color:red;'>onRef父组件调用子组件的方法和数据</span>

```js
class RefEg extends React.Component{
    constructor(props){
        super(props);
    }
    componentDidMount(){
        //在组件挂载阶段结束前，是不能获取到ref的，这应尤为的注意
        console.log(this.refs);
        this.refs.spanRef.innerHTML='hahh';
    }
    backref=(ref)=>{
        this.welcome=ref;
    }
    /*父调子第3步*/
    dadCallBaby=()=>{
        this.welcome.test();
    }
    render(){
        return(
            <div>
                <span style={{width:'100px',height:'50px'}} ref='spanRef'>这是对span Dom节点的引用</span>
                 {/*父调子第1步*/}
                <Welcome ref='welcomeRef' onRef={this.backref}/>
                <button onClick={this.dadCallBaby}></button>
            </div>
        )
    }
}
```
```js
class Welcome extends React.Component{
    constructor(props){
        super(props);
    }
    componentDidMount(){
        {/*父调子第2步*/}
        this.props.onRef(this);
    }
    test=()=>{
        console.log('父组件调用子组件方法成功！！！')
    }
    render(){
        return(
            <div>
               <h1>这里是对组件实例的引用</h1>
            </div>
        )
    }
}
```

# 5. React进阶

## 5.1 v16.8.6——[最新生命周期](https://react.docschina.org/docs/react-component.html)

1. 挂载阶段

       SAFE: 
       constructor(props) 
       static getDerivedStateFromProps(props, state)  注：state 的值在任何时候都取决于 props 时
       render() 
       componentDidMount()
    
       UNSAFE:
       componentWillMount()

2. 更新阶段

        SAFE：
        static getDerivedStateFromProps() 
        shouldComponentUpdate() 
        getSnapshotBeforeUpdate(prevProps, prevState) 
    
        componentDidUpdate(prevProps, prevState, snapshot)
    
        UNSAFE：
        componentWillUpdate()  
        componentWillReceiveProps()

3. 卸载阶段

        SAFE:
        componentWillUnmount()

4. 错误处理函数

当渲染过程，生命周期或子组件的构造函数中抛出错误时，会调用如下方法

    static getDerivedStateFromError() （更新阶段）
    componentDidCatch()

## 5.2 [高阶组件--HOC](https://zh-hans.reactjs.org/docs/higher-order-components.html)

    高阶组件是参数为组件，返回值为新组件的函数。
    
    组件是将 props 转换为 UI，而高阶组件是将组件转换为另一个组件。
# 6 其他

`React.FC< >`