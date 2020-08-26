# 面试

[web前端面试100题](<https://zhuanlan.zhihu.com/p/82124513>)

1. 跨域是指浏览器允许向服务器发送跨域请求，从而克服Ajax只能**同源**使用的限制。同源（协议+ip+端口），否则为跨域。

   - **jsonp**的原理就是利用`<script>`标签没有跨域限制，通过`<script>`标签src属性，发送带有callback参数的GET请求，服务端将接口返回数据拼凑到callback函数中，返回给浏览器，浏览器解析执行，从而前端拿到callback函数返回的数据。

   jsonp的缺点：只能发送get一种请求。

   ```js
   //vue axios 的jsonp
   this.$http = axios;
   this.$http.jsonp('http://www.domain2.com:8080/login', {
       params: {},
       jsonp: 'handleCallback'
   }).then((res) => {
       console.log(res) 
   })
   ```

   - 跨域资源共享（CORS)，主要就是通过设置Access-Control-Allow-Origin来进行的

   - nginx代理跨域

   - node.js中间件

2. webpack：是一个模块打包工具，管理模块依赖，并编译输出模块所需的静态资源。它有自己的配置文件，更好的语法兼容

3. Json与XML的区别：

   - Json：体积小，解析快，传输快
   - XML：数据的描述性更好

4. tcp的三次握手和四次挥手，

   连接建立：1发送端发syn/ack，2接收端收到返回ack，3发送端发送ack表示握手结束。

   断开连接：1主动关闭方发送fin告诉被动关闭方，我不会发数据给你了，2被动关闭方收到后，发送ack表示确认，3被动关闭方发送fin，告诉主动关闭方，我不会发数据给你了，4主动关闭方收到后，发送ack表示确认。

5. TCP和UDP的区别：TCP安全可靠，适于大量数据传输，对可靠性要求高的传输环境，而UDP恰恰相反，但传输快。

6. 创建ajax过程：

   - 创建XMLHttpRequest异步调用对象，
   - 创建Http请求，设置请求方法，url等信息
   - 设置响应Http请求状态变化的函数
   - 发送Http请求
   - 获取异步调用返回的数据
   - 数据变化响应至视图

7. 渐进增强和优雅降级

   - 渐进增强：针对低版本浏览器，保证基本功能，然后在针对高级浏览器增强用户良好体验度
   - 优雅降级：一开始就构建完整的功能，然后在针对低版本浏览器进行兼容

8. 常见web安全及防护原理

   - sql注入原理：就是通过把SQL命令插入到Web表单递交或输入域名或页面请求的查询字符串，最终达到欺骗服务器执行恶意的SQL命令。
   - Xss(cross-site scripting)攻击指的是攻击者往Web页面里插入恶意 html标签或者javascript代码。

9. XSS是获取信息，不需要提前知道其他用户页面的代码和数据包。CSRF（Cross-site request forgery，跨域请求伪造）是代替用户完成指定的动作，需要知道其他用户页面的代码和数据包（登录受信任的a，在不登出a下，访问危险b）。

10. websocket是html5的协议，保持客户端与服务器之间的双工通信，他们之间的连接是持久的。http是非持久的，只通过发送请求得到返回方式通信。

11. worker 线程

    ```js
    创建线程 ：worker= new Worker("url");
    向线程发送数据：worker.postMessage(data);
    监听并处理向线程发来的数据：worker.onmessage=function(event){ event.data}
    ```

    - 异步是一种技术功能要求，多线程是实现异步的一种手段。除了使用多线程可以实现异步，异步I/O操作也能实现。

12. Http和Https：HTTP协议通常承载于TCP协议之上，在HTTP和TCP之间添加一个安全协议层（SSL或TSL），这个时候，就成了我们常说的HTTPS。

13. AMD、CMD、CommonJs、ES6的对比：他们都是用于在模块化定义中使用的，AMD（异步）、CMD、CommonJs（nodejs中采用，同步）是ES5中提供的模块化编程的方案，import/export是ES6中定义新增的

14. Javascript垃圾回收方法：标记清除与引用计数（循环引用）

15. 性能优化

    - 少用全局变量，减少Dom操作，多变量声明合并

16. defer和async：

    - defer：页面载入完毕，解析完毕才异步执行\<script/>
    - async：启动新线程，异步执行

17. 设计模式： 工厂模式（解决重复实例化的问题），构造函数模式（既解决重复实例化的问题 ，又解决了对象识别），单例模式

18. cookie：

    - 在同一个域名下，cookie有数量限制，大小最大4kb左右，如果被拦截将出现安全问题。
    - cookie过期时间，不会永远生效

19. webstorage本地存储：

    - locakStorage：用于持久化的本地存储，除非主动删除数据，否则数据是永远不会过期的。

    - sessionStorage：用于本地存储一个会话（session）中的数据，这些数据只有在同一个会话中的页面才能访问并且当会话结束后数据也随之销毁。浏览器关闭后自动删除

      ```js
      localStorage.setItem("lastname", "Smith");
      localStorage.getItem("lastname");
      localStorage.removeItem("key");
      ```

20. cookie与webStorage的区别：

    - cookie的作用是与服务器进行交互，作为HTTP规范的一部分而存在 ，而Web Storage仅仅是为了在本地“存储”数据而生。
    - cookie需要自行封装，而storage提供相应的方法去获取和设置
    - 多标签页之间可以通过cookie和storage

21. cookie和session的区别：

    - cookie数据存放在客户的浏览器上，session数据放在服务器上。
    - session会在一定时间内保存在服务器上。当访问增多，会比较占用你服务器的性能

22. display:none和visibility:hidden的区别：

    - 前者隐藏元素，收紧长宽，释放空间
    - 后者隐藏元素，空间仍保留

23. link与@import的区别：

    - link属于HTML标签，而@import是CSS提供的;
    - 页面被加载的时，link会同时被加载，而@import被引用的CSS会等到引用它的CSS文件被加载完再加载;

24. box-sizing：content-box(默认)/border-box;

25. > > > > !important>内联样式 > ID 选择器 > 类选择器 = 属性选择器 = 伪类选择器 > 标签选择器 = 伪元素选择器

26. 语义化：失去样式但有清晰的结构，易于网页检索，爬虫抓取，可读性增强

27. html5中只有\<!DOCTYPE html>，请始终向 HTML 文档添加 <!DOCTYPE> 声明，这样浏览器才能获知文档类型。html4支持Strict、Transitional，FrameSet

28. HTML5 现在已经不是 SGML 的子集，主要是关于图像（audio，vedio），位置（location），存储（storage），多任务，语义化等功能的增加。

29. 无样式内容闪烁：引用CSS文件的@import就是造成这个问题的罪魁祸首，在head里面通过\<link>和或者\<script>元素就可以了。

30. undefined 与 null ：声明但未赋初值，null表示未存在的对象，通常用作对象的垃圾回收，

    - undefined：
      - 声明但未赋初值
      - 调用函数时，应该提供的参数没有提供（函数未传参）
      - 访问对象里没有的属性
      - 函数没有返回值，默认值为undefined
    - null：
      - 通常用作对象的垃圾回收

31. js延迟加载：

    - defer和async
    - 动态创建script DOM，加载后callback，

32. call()和apply()

    - ECMAScript 规范给所有函数都定义了apply 和 call 方法

    - ```js
      //apply()
      //apply()方法传入两个参数,一个是函数上下文的对象，一个是该函数传入的数组形式的参数
      var Person = {
         name :'Lucy'   
       }
      function getName(name1,name2){
         console.log(this.name + name1 + name2);    //Lucy Tom Tim
      }
      getName.apply(Person,['Tom','Tim']);  
      //第一个参数指向person对象 ，this.name 指向 person.name ; 第二个参数是一个数组，数组中的元素分别代表函数的参数一和参数二
      ```

    - ```js
      //call()
      // call()方法传入两类参数，第一个参数是上下文对象，第二类参数以列表的形式传入
      var Person = {
         name :'Lucy'   
       }
      function getName(name1,name2){
         console.log(this.name + name1 + name2); //Lucy Tom Tim
      } 
      getName.call(Person,'Tom','Tim');
      ```

    - 两个方法的作用:

      - 改变this指向
      - 借用别的对象的方法
      - 调用函数

33. 内存泄漏：指任何对象在您不再拥有或需要它之后仍然存在。造成原因：对象循环引用，定时器忘回收，DOM 引用，闭包中的变量

    - 垃圾回收机制：引用计数和标记清除

34. 同源限制：黑客通过iframe嵌入银行页面，获取账户密码。

35. **GET和POST**：

    - GET：通过URL传值，对参数大小也有2000个字符的限制，需要使用Request.QueryString来取得变量的值
    - POST：发送大量数据，并且更安全稳定，需要使用Request.Form来获取变量的值，

36. 事件：

    - 我们在页面上的某个操作（有些操作对应多个事件）
    - 事件机制：捕获阶段和冒泡阶段（ie只有冒泡阶段），阻止冒泡ev.stopPropagation()

37. 服务器主动推数据到客户端：websocket，SSE( server send event )， Commet

38. 网站重构：在不改变UI的情况下，对网站进行优化。

39. mongoDB和MySQL：

    - 前者非关系型数据库，对海量数据有明显的优势
    - 后者为关系型数据库

40. Promise 对象用来进行延迟(deferred) 和异步(asynchronous ) 计算。

    ```js
    let data={...}
    function promiseFun(params){
       let p =new Promise((resolve,reject)=>{
            if(succed){
                ...
                resolve(res)
            }else{
                ...
                reject(res)
            }
    	}) 
       return p
    }
    promiseFun(data).then((res)=>{/*success*/},(res)=>{/*fail*/})
    
    ```

    

41. 事件代理：事件代理”即是把原本需要绑定在子元素的响应事件（click、keydown......）委托给父元素，让父元素担当事件监听的职务。事件代理的原理是DOM元素的事件冒泡。**可以大量节省内存占用，减少事件注册**

42. 栈内存与堆内存：

    - 栈内存：运行效率高，内存小。用于存放基本数据类型。
    - 堆内存：用于存放复合类型数据（引用数据类型），效率低，内存大
    - 栈内存中变量一般在它的当前执行环境结束就会被销毁被垃圾回收制回收， 而堆内存中的数据当没有变量引用时，就会被GC。
    - 闭包中的变量并不保存中栈内存中，而是保存在堆内存中。 这也就解释了函数调用之后之后为什么闭包还能引用到函数内的变量。

43. 深拷贝与浅拷贝：

    - 浅拷贝：引用类型数据传址

      - 直接赋值（针对于引用数据类型，将堆内存的地址赋值给了新变量，变化会同步执行）

      - assign：第一层伪深拷贝

    - 深拷贝：引用类型数据传值（新建堆内存）

      - 递归赋值：
      - JSON.parse()和JSON.stringify()

44. 闭包：是一种保护私有变量的机制，在函数执行时形成私有的作用域，保护里面的私有变量不受外界干扰。直观的说就是形成一个不销毁的栈环境。

45. 作用域链：当前作用域没有定义的变量，叫做自由变量，函数访问自由变量，是从祖辈作用域中一层层搜索，这样就叫做作用域链。

46. 盒模型：

    - 盒模型本质上是一个盒子，封装周围的HTML元素，它包括：边距，边框，填充，和实际内容。
    - 行内模型：不可设置长宽，可以与其他元素位于同一行
    - 块模型：可设长宽，不可与其他元素位于同一行

47. let，const ，var：

    - let和const：存在块作用域内，不存在变量提升
    - var：存在于函数体和全局声明，存在变量提升，var声明的变量会挂载在window，而let和const声明的变量不会，
    - const：声明后不能再修改，如果声明的是复合类型数据，可以修改其属性
    - **ES6 之前 JavaScript 没有块级作用域,只有全局作用域和函数作用域**。ES6 的到来，为我们提供了‘块级作用域’,可通过新增命令 let 和 const 来体现。

48. ：

49. 

    

# 数组

```javascript
//切合操作
1. array1.concat(array2,array3,...,arrayX)//衔接，多个数组为一个数组
2. array.slice(start, end)//切割，数组的部分元素，左闭右开

//条件操作
3. array.every(function(currentValue,index,arr), thisValue)//是否每个都满足，所有值返回为true，函数才返回true
4. array.some(function(currentValue,index,arr),thisValue)//是否部分满足,有一个返回为true就行，函数才返回true

5. array.fill(value, start, end)//把数组的部分元素填充为指定value

//遍历操作
6. array.map(function(currentValue,index,arr), thisValue)//遍历数组所有元素，按回调函数的方式返回相应的数组元素
7. array.filter(function(currentValue,index,arr), thisValue)//筛选数组元素，将返回ture的item重新组成新的数组。
8. array.forEach(function(currentValue, index, arr), thisValue)//遍历数组
9. Array.from(object, mapFunction, thisValue)//from() 方法用于通过拥有 length 属性的对象或可迭代的对象来返回一个数组。

//查找判断
10. array.find(function(currentValue, index, arr),thisValue)//返回第一个满足条件的数组元素
11. array.findIndex(function(currentValue, index, arr), thisValue)//返回第一个满足条件数组元素的索引
12. array.indexOf(item,start)//查找目标元素第一次出现的位置
13. array.lastIndexOf(item,start)//查询数组中目标元素的索引//对原数组的操作
14. array.includes(searchElement, fromIndex)//判断数组中是否包含某个元素

11. Array.isArray(obj)//判断是否为obj是否为数组
12. array.join(separator)//以separator的为分割，链接成字符串

//对数组本身的操作
14. array.pop()//弹出数组最后一个元素
15. array.push(item1, item2, ..., itemX)//压入多个元素到数组末尾
16. array.shift()//弹出数组第一个元素
17.	array.unshift(item1,item2, ..., itemX)//压入多个元素到数组首部

//顺序操作
18. array.sort()//排序，默认排序顺序为按字母升序。
				//对于数字来说，这种按字母排序的方式，会导致40会在5前面。 数字升序排列arr3.sort(function(a,b){return a-b})
19. array.reverse()//逆序

20. array.reduce(function(total, currentValue, currentIndex, arr), initialValue)//接收一个函数作为累加器,最终返回累加和
19. array.toString()
20. 
```



# 字符串

```js
string.replace(searchvalue,newvalue)
string.match(regexp)//返回存放匹配结果的数组
string.indexOf(searchvalue,start)
string.lastIndexOf(searchvalue,start)
string.substring(from, to)//左闭右开
string.substr(start,length)
```

# 全局函数

```js
parseInt(string, radix)
parseFloat(string)
```

# Window对象



# 项目

## 滚动到框的最低端

```js
function toButtom(){
   let chatbox=this.$refs.chatbox
	chatbox.scrollTop=chatbox.scrollHeight- chatbox.clientHeight; 
}

```

## socket.io

是websocket的一种实现，它结合了其他东西

```bash
npm install --save socket.io-client#客户端安装
npm install --save socket.io#服务器端安装
```

webSocket

WebSocket同HTTP一样也是应用层的协议，但是它是一种双向通信协议，是建立在TCP之上的。

Socket位于传输控制层与应用层之间的一组接口。

HTTP协议是非持久化的，单向的网络协议，在建立连接后只允许浏览器向服务器发出请求后，服务器才能返回相应的数据。

```js
let websocket=new WebSocket("url");
websocket.onmessage //当WebSocket接收到远程服务器的数据时触发该事件
websocket.onopen //当WebSocket建立网络连接时触发该事件
websocket.send() //向远程服务器发送消息
websocket.close() //关闭websocket
websocket.onerror //当网络出现连接错误时触发该事件
websocket.onclose //当网络被关闭时触发该事件

```

## nginx

# React

1. 生命周期：挂载阶段，更新阶段，卸载阶段

   挂载阶段：这个阶段组件被创建，执行初始化，并被挂载到DOM中，完成组件的第一次渲染。

   - constructor()，用于初始化state和绑定方法，
   - static getDerivedStateFromProps()：此方法适用于[罕见的用例](https://zh-hans.reactjs.org/blog/2018/06/07/you-probably-dont-need-derived-state.html#when-to-use-derived-state)，即 state 的值在任何时候都取决于 props。
   - componentWillMount()，挂载前，这里不能操作dom对象，即将过时，
   - render()，这是定义组件时<span  style='color:red;'>唯一必要的方法</span>
   - componentDidMount()，挂载后，依赖于 DOM 节点的初始化应该放在这里。

   更新阶段：组件被挂载到DOM后，组件的props或state可以引起组件更新。

   - componentWillReceiveProps(nextProps)——porps改变会执行次函数，即将过时
   - shouldComponentUpdate(nextProps,nextState)——props改变后或state改变都会执行此生命周期，return true执行更新，返回false不执行更新操作
   - componentWillUpdate(nextProps,nextState)——更新前，即将过时
   - render()——渲染
   - componentDidUpdate(prevProps,prevState)——更新后

   卸载阶段：组件在被卸载前调用

   - componentWillUnmount：通常用来清除组件中使用的定时器，恢复数据仓库中的初始数据参数。

2. diff策略：

   - tree diff：对树分层比较（层级比较），两棵树 只对**同一层次节点** 进行比较（跨层就涉及到重新创建和删除操作）。如果该节点不存在时，则该节点及其子节点会被完全删除，不会再进一步比较。
   - component diff：
     - 同类型两个组件
       - 按照层级比较虚拟DOM
       - 从A变到B，通过shouldComponent()判断是否需要更新
     - 不同类型组件：替换组件
   - element diff：当节点处于同一层级时，diff提供三种节点操作：**删除、插入、移动**。
     - 新节点就插入，少节点就删除
     - 移动：渲染后的index>渲染前的index就移动（最后一个节点移动到第一个节点，将会导致性能降低）

3. 

   

# 简单编程

```js
//利用闭包做一个计数器
let add=(function(){
    let count=0;
    return function(){
        count+=1
        return count
    }
})()//第一个括号用于定义外函数，第二个括号用于执行一次外函数（外函数只执行了一次），返回一个内函数给add变量
console.log(add())
console.log(add())
console.log(add())

//数组去重
```

