# VueJS

# 1 初识vue

## 1.1 设计模式

### MVC

Model-View-Control，数据存储（变量）-用户视图（界面）-控制器（事件交互）

![MVC.PNG](/legend/MVC.PNG)

### MVVM

本质上就是MVC 的改进版，Model-View-ViewModel(负责业务逻辑处理，对数据加工后给视图展示），它可以取出 Model 的数据同时帮忙处理 View 中由于需要展示内容而涉及的业务逻辑。

![MVC.PNG](/legend/MVVM.JPG)

## 1.2 三大前端框架

VueJS(70%)，ReactJS(20%)，AngularJS(10%)

VueJS：能够帮我们减少不必要的DOM操作(虚拟DOM)，提高渲染效率，双向数据绑定(数据驱动）。

### 1.2.1 Vue和React的相同点

- 虚拟DOM
- 轻量级
- 响应式组件
- 支持服务器渲染
- 易于集成路由工具、打包工具以及状态管理工具

注：Vue在国内很受欢迎；React在国内外都很受欢迎，适合做大型网站

**服务器端渲染**:页面在后端将html拼接好的然后将之返回给前端完整的html文件，浏览器拿到这个html文件之后就可以直接解析展示了。

**客户端渲染**：**ajax的兴起**，使得业界就开始推崇**前后端分离**的开发模式，即后端不提供完整的html页面，而是提供一些api使得前端可以获取到json数据，然后前端拿到json数据之后再在前端进行html页面的拼接，然后展示在浏览器上。

### 1.2.2 极简入门

```html
<!DOCTYPE html>
<html>
    <head>
        <title>
          Vuejs
        </title>
        <meta charset="utf-8"/>
        <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
    </head>
    <body>
        <!-- VIEW -->
        <div id="app">
            {{ message }}
        </div>
       
        <style>
            #app{
                font-size:2em;
                color:red;
            }
        </style>
        <script type="text/javascript">
          console.log(Vue);
        //   VIEW-MODEL
        //   Vue
          let app = new Vue({
                        el: '#app',
                        // MODEL
                        data: {
                            message: 'Hello Vue!'
                        }
                    })
        </script>
    </body>
</html>

```

![vueapp.png](/legend/vueapp.png)

## 1.3 模板语法

```vue
<template>
    <div>
       <div>{{message}}</div>
       <div :class="light">你好，世界</div>
    </div>
<template/>
<style scoped>
    .lightstyle{
        color:#f00;
        font-size:20px;
        font-weight:700;
    }
</style>
<script>
    export default{
        data(){
           light:'lightstyle',
           message:'hello，Vue.js'
        },
    }    
</script>
```

# 2 Vue基础



## 2.1 条件与列表渲染

v-if、v-else、v-else-if，v-show

v-show 就简单得多——不管初始条件是什么，元素总是会被渲染，并且只是简单地基于 CSS 进行display切换。

v-if 有更高的切换开销，而 v-show 有更高的初始渲染开销。因此，如果需要非常频繁地切换，则使用 v-show 较好；如果在运行时条件很少改变，则使用 v-if 较好。

```vue
<template>
	<!--template下只能有一个根元素-->
    <div>
       <div id="conditionView">
            <!--双引号里面只能有一个js语句-->
            <div v-if="type === 'A'">
              A
            </div>
            <div v-else-if="type === 'B'">
              B
            </div>
            <div v-else-if="type === 'C'">
              C
            </div>
            <div v-else>
              Not A/B/C
            </div>
        </div>
        <div id="loopView">
            <div v-for="item in list">
                <span>{{item.text}}</span>
    		</div>
    	</div>
    </div>
<template/>
<script>
	export default{
        //组件的data必须是一个函数
        data(){
            return{
                type:"A",
                list:[
                    {text:"李欣"},
                	{text:"理性"},
                    {text:"比心"},
                    {text："得之我幸，失之我命"}
                ]
            }
        }
        methods:{
        
    	}
    }
</script>
<style>
    
</style>
```

## 2.2 计算属性与侦听器

1. **计算属性**：是放在data外的另一种属性，computed里面的属性是依赖于其它属性进行计算的，一旦其他属性（**多个属性**）发生变化，他就会触发计算属性的改变，计算属性默认getter方法，如果需要setter方法需要自己定义。computed里的属性是在挂载阶段执行的（介于beforeMount和mounted两个钩子函数之间）
2. **侦听器**：一个侦听器只能监听某一个属性的变化，而且属性需要在data中注册，在watch中键为需要监听的属性，值为data中属性发生变化后需要做的逻辑处理

```vue
<template>
	<div>
         <h3>计算属性</h3>
         <!-- 可以在console上改变属性值，看看效果 -->
         <div>{{cfullname}}</div>
         <h3>侦听属性</h3>
         <div>{{wfullname}}</div>
    </div>
</template>
<script>
    export default {
        name:"computedandwatch",
        data(){
             wfirstname:"li",
             wlastname:"xin",
             wfullname:"lixin",
        },
        computed:{//计算属性，是放在data外的另一种属性
                  //这里面的属性，是依赖于其它属性进行计算的，
                  //一旦其他属性发生变化，他就会触发计算属性的改变
                  //计算属性默认getter方法，如果需要setter方法需要自己定义
            cfullname:{
                get:function () {     
                    let setter=this.cfirstname + ' ' + this.clastname;
                    console.log("计算属性",setter)
                    return setter
                },
                set:function(val){
                    return "秦"+val
                }
            }
         },
        watch:{//侦听器
            //一旦监听到data的对应属性发生变化，就会触发它对应的方法
            wfirstname: function (val) {
                this.wfullname = val + ' ' + this.wlastname
            },
            wlastname: function (val) {
                this.wfullname = this.wfirstname + ' ' + val
            }
         },
    }
</script>
```



## 2.3 Class 与 Style 绑定

操作元素的 class 列表和内联样式是数据绑定的一个常见需求。因为它们都是 attribute，所以我们可以用 `v-bind` 处理它们：只需要通过表达式计算出字符串结果即可。不过，字符串拼接麻烦且易错。因此，在将 `v-bind` 用于 `class` 和 `style` 时，Vue.js 做了专门的增强。表达式结果的类型除了字符串之外，还可以是对象或数组。

用在组件上：当在一个自定义组件上使用 `class` property 时，这些 class 将被添加到该组件的根元素上面。这个元素上已经存在的 class 不会被覆盖。

```vue
<template>
	<div>
        <div v-bind:class="{ active: isActive,}">李欣</div><!--条件对象class绑定-->
    	<div v-bind:class="classObject">你好</div><!--对象class绑定-->
        <div v-bind:class="[activeClass, errorClass]">笔芯</div><!--数组class绑定,数组语法中也可以使用对象语法-->
        <div class="static" v-bind:class="{ active: isActive, 'text-danger': hasError }">
    		对象中传入更多字段来动态切换多个 class。此外，v-bind:class 指令也可以与普通的 class attribute 共存
    	</div>
        
        <div v-bind:style="{ color: activeColor, fontSize: fontSize + 'px' }"></div><!--对象style绑定-->
        <div v-bind:style="[baseStyles, overridingStyles]">数组语法可以将多个样式对象应用到同一个元素上</div>
    </div>
	
</template>
<script>
    export default{
        data(){
            return {
                isActive:true,
                error:false,
                activeClass: 'active',
  				errorClass: 'text-danger',
                activeColor:'#f00',
                fontSize:20,
                baseStyles:{
                    color:'#f00',
                    fontSize:'10px'
                },
                overridingStyles:{
                    width:"100px",
                    height:"50px"
                }
            }
        },
        computed:{
            classObject:function(){
               active: this.isActive && !this.error,
      		  'text-danger': this.error
            }
        }
    }
</script>
```



## 2.4 事件处理

可以用 `v-on` 指令监听 DOM 事件，并在触发绑定的自定义事件处理函数。

有时也需要在内联语句处理器中访问原始的 DOM 事件。可以用特殊变量 `$event` 把它传入方法。

### 事件处理修饰符

修饰符可以串联，使用修饰符时，顺序很重要

eg： `v-on:click.prevent.self` 会阻止**所有的点击**，而 `v-on:click.self.prevent` 只会阻止对元素自身的点击。

- `.stop`，阻止事件继续传播
- `.prevent`，阻止事件继续传播
- `.capture`，事件捕获阶段
- `.self`，限定事件源自自身元素
- `.once`，事件只触发一次
- `.passive`，滚动行为将会立即触发

### 按键

#### 按键修饰符

在监听键盘事件时，我们经常需要检查详细的按键。Vue 允许为 `v-on` 在监听键盘事件时添加按键修饰符

- `.enter`
- `.page-down`

#### [按键码](<https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/keyCode>)

```html
<div v-on:keyup.13="dosomething"></div>
```

#### 系统修饰符

- `.ctrl`
- `.alt`
- `.shift`
- `.meta`

#### 鼠标按钮修饰符

- `.left`
- `.right`
- `.middle`

```vue
<template>
	<div>
        <button v-on:click="warn('Form cannot be submitted yet.', $event)">
  			Submit
		</button>
    </div>
</template>
<script>
    export default{
        data(){
            return{
                
            }
        },
        methods:{
            warn: function (message, event) {
    			// 现在我们可以访问原生事件对象
                if (event) {
                    event.preventDefault()
                }
                alert(message)
            }
        }
    }
</script>
```



## 2.5 双向数据绑定

你可以用 `v-model` 指令在表单 `<input>`、`<textarea>` 及 `<select>` 元素上创建双向数据绑定。

`v-model` 在内部为不同的输入元素使用不同的 property 并抛出不同的事件

v-model的数据修饰符：

- `.lazy`，`change` 事件_之后_进行同步
- `.number，数据类型转换
- `.trim`，删掉数据首尾空格

```html
<input v-model.lazy.trim.number="age" type="number">
```

## 2.6 过渡动画

Vue 提供了 `transition` 的封装组件，在下列情形中，可以给任何元素和组件添加进入/离开过渡

- 条件渲染 (使用 `v-if`)
- 条件展示 (使用 `v-show`)
- 动态组件
- 组件根节点

```vue
<template>
	<div>
         <button v-on:click="show = !show">测试过渡效果</button>
            <div :style="{display:'flex'}">
                <transition name="fade">
                    <img  v-if="show" :src="obj.img" class="trans" />
        		</transition>
                <transition name="slide">
                    <img  v-if="show" :src="obj.img" class="trans" />
        		</transition>
                <div class="trans"></div>
        </div>
    </div>
</template>
<script>
    export default{
        data(){
            return{
                show:true,
                obj:{
                    name:"李欣",
                    music:"100分",
                    comment:"优秀如斯",
                    img:"./legend/lixin.png",
                    size:10
                },
            }
        },
        methods:{
          
        }
    }
</script>
<style scoped>
     .trans{
		 height:10vw;
      }
    .fade-enter-active, .fade-leave-active {
        transition: opacity 2s;
    }
    .fade-enter, .fade-leave-to /* .fade-leave-active below version 2.1.8 */ {
        opacity: 0;
    }
    .slide-enter-active, .slide-leave-active {
        transition: transform 2s;
    }
    .slide-enter, .slide-leave-to {
        transform:translateX(200px)
    }
</style>
```



## 2.7 [生命周期](<https://segmentfault.com/a/1190000011381906>)

vue实例的一般键

```json
{
    el:"#app",
    template:""
    data:{},
	props:[],
	methods:{},
	components:{},
	computed:{},
	watch:{},
	beforeCreate(){},
	created(){},
	render(createElement) {
        return createElement('h1', 'this is createElement')
    },
	beforeMount(){},
	mounted(){},
	beforeUpdate(){},
	updated(){},
	beforeDestroy(){},
	destroyed(){}
}
```



1. new Vue()：构造函数生成Vue实例。

2. [init Events&LifeCycle](<https://blog.csdn.net/qq_40542728/article/details/103733037>)：

     - 初始化实例属性是第一步，Vue.js通过**initLifecycle**函数向Vue实例中挂载属性，该函数接收Vue实例作为参数。
     - 初始化事件是指将父组件在模板中使用的v-on注册的事件添加到子组件的事件系统（Vue.js的事件系统）中。Vue.js通过**initEvents**函数执行初始化事件相关的逻辑。

3. [init injection&reactivity](<https://blog.csdn.net/jifukui/article/details/106807694>):

     - 在高等级的组件中使用用于父组件向子组件传递数据。祖先组件在provide中提供后代可使用的数据，后代组件在inject中设置使用祖先组件的属性名。

4. **在beforeCreate和created钩子函数之间的生命周期**
     在这个生命周期之间，进行初始化事件，进行数据的观测，可以看到在created的时候数据已经和data属性进行绑定（放在data中的属性当值发生改变的同时，视图也会改变）。注意看下：此时还是没有el选项

5. **created钩子函数和beforeMount间的生命周期**

   首先会判断对象是否有el参数。如果有的话就继续向下编译，如果没有el选项，则停止编译，也就意味着停止了生命周期，直到在该vue实例上调用vm.$mount(el)。此时注释掉代码中:el: '#app'（挂载点，找不到挂载点，就不会执行beforeMount），然后运行可以看到到created的时候就停止了。如果没有el选项，则停止编译，也就意味着停止了生命周期，直到在该vue实例上调用vm.$mount(el)。如果我们在后面继续调用vm.$mount(el),可以发现代码继续向下执行了

   然后，我们往下看，template参数（）选项的有无对生命周期的影响。
   （1）.如果vue实例对象中有template参数选项，则将其作为模板编译成render函数。
   （2）.如果没有template选项，则将外部HTML作为模板编译。
   （3）.可以看到template中的模板优先级要高于outer HTML的优先级。

   render函数选项 > template选项 > outer HTML.

   ```html
   <!DOCTYPE html>
   <html lang="en">
   <head>
     <meta charset="UTF-8">
     <meta name="viewport" content="width=device-width, initial-scale=1.0">
     <meta http-equiv="X-UA-Compatible" content="ie=edge">
     <title>vue生命周期学习</title>
     <script src="https://cdn.bootcss.com/vue/2.4.2/vue.js"></script>
   </head>
   <body>
     <div id="app">
       <!--html中修改的-->
       <h1>{{message + '这是在outer HTML中的'}}</h1>
     </div>
   </body>
   <script>
     var vm = new Vue({
       el: '#app',
       template: "<h1>{{message +'这是在template中的'}}</h1>", //在vue配置项中修改的
       data: {
         message: 'Vue的生命周期'
       }
   </script>
   </html>
   
   
   ```

6. **beforeMount和mounted 钩子函数间的生命周期**

   可以看到此时是给vue实例对象添加$el成员(子元素，子实例），并且替换掉挂在的DOM元素。因为在之前console中打印的结果可以看到beforeMount之前el上还是undefined。

   在mounted之前h1中还是通过{{message}}进行占位的，因为此时还没有挂在到页面上，还是JavaScript中的虚拟DOM形式存在的。在mounted之后可以看到h1中的内容发生了变化。

7. beforeUpdate钩子函数和updated钩子函数间的生命周期

   当vue发现data中的数据发生了改变，会触发对应组件的重新渲染，先后调用beforeUpdate和updated钩子函数

8. beforeDestroy和destroyed钩子函数间的生命周期

   beforeDestroy钩子函数在实例销毁之前调用。在这一步，实例仍然完全可用。
   destroyed钩子函数在Vue 实例销毁后调用。调用后，Vue 实例指示的所有东西都会解绑定，所有的事件监听器会被移除，所有的子实例也会被销毁。

9. beforeDestroy和destroyed钩子函数间的生命周期

   beforeDestroy钩子函数在实例销毁之前调用。在这一步，实例仍然完全可用。
   destroyed钩子函数在Vue 实例销毁后调用。调用后，Vue 实例指示的所有东西都会解绑定，所有的事件监听器会被移除，所有的子实例也会被销毁。

![lifecycle.png](legend/lifecycle.png)

```vue

```

```vue
<template>
	<div>
        <button v-on:click="warn('Form cannot be submitted yet.', $event)">
  			Submit
		</button>
    </div>
</template>
<script>
    export default{
        data(){
            return{
                
            }
        },
        methods:{
            warn: function (message, event) {
    			// 现在我们可以访问原生事件对象
                if (event) {
                    event.preventDefault()
                }
                alert(message)
            }
        },
        beforeCreate(){
            console.log("实例创建之前",this)
            //下面输出全为undefined
            console.log("%c%s", "color:red" , "el     : " + this.$el);
            console.log("%c%s", "color:red","data   : " + this.$data);
            console.log("%c%s", "color:red","message: " + this.message); 
        },
        created(){
            console.log("实例创建之后",this)
            console.log("%c%s", "color:red" , "el     : " + this.$el);//undefined
            console.log("%c%s", "color:red","data   : " + this.$data);//已被初始化
            console.log("%c%s", "color:red","message: " + this.message);//已被初始化
        },
        beforeMount(){
            console.log("挂载之前",this)
            console.log("%c%s", "color:red" , "el     : " + this.$el);
            console.log("%c%s", "color:red","data   : " + this.$data);
            console.log("%c%s", "color:red","message: " + this.message); 
        },
        mounted(){
            console.log("挂载之后",this)
            console.log("%c%s", "color:red" , "el     : " + this.$el);
            console.log("%c%s", "color:red","data   : " + this.$data);
            console.log("%c%s", "color:red","message: " + this.message); 
        },
        beforeUpdate(){
            console.log("更新页面之前")
        },
        updated(){
            console.log("更新页面之后")
        },
        beforeDestroy(){
            console.log("页面销毁之前")
        },
        destroyed(){
            console.log("页面销毁之后")
        }
    }
</script>
```



## 2.8 [VueCli](<https://cli.vuejs.org/zh/guide/>)

下载安装nodejs（javascript）运行环境，并安装，我们的Vuecli需要在此环境中使用。

CLI (`@vue/cli`) 是一个全局安装的 npm 包，提供了终端里的 `vue` 命令。它可以通过 `vue create` 快速搭建一个新项目，或者直接通过 `vue serve` 构建新想法的原型（单个 `*.vue` 文件）。你也可以通过 `vue ui` 通过一套图形化界面管理你的所有项目。我们会在接下来的指南中逐章节深入介绍。

```bash
npm install -g @vue/cli
#进入项目存放文件夹，cmd
#创建新项目
vue create 项目名
#然后选择一些简单的初始依赖环境配置

Vue CLI v4.4.6
? Please pick a preset: (Use arrow keys)
> default (babel, eslint)
Manually select features#将>上下选择移至此项，然后回车，手动选择相关初始配置

? Please pick a preset: Manually select features
? Check the features needed for your project: (Press <space> to select, <a> to toggle all, <i> to invert selection)
>(*) Babel#编译
 ( ) TypeScript
 ( ) Progressive Web App (PWA) Support
  ( ) Router#路由
 ( ) Vuex#状态管理
 ( ) CSS Pre-processors
 ( ) Linter / Formatter
  ( ) Unit Testing
 ( ) E2E Testing
 #只选择babel
 
? Where do you prefer placing config for Babel, ESLint, etc.? (Use arrow keys)
> In dedicated config files
  In package.json#将>上下选择移至此项，然后回车
  
? Save this as a preset for future projects? (y/N) y #是否保存此项目的配置作为未来新创项目的初始配置，依据你的意愿选择y或n
? Save preset as:未来初始配置的名字

$ cd 项目文件名
$ npm run serve
```

![vuegate.png](E:/note/Vue/legend/vuegate.png)

- public/index.html——入口文件

# 3 组件

我们可以将组件进行任意次数的复用。**一个组件的 data 选项必须是一个函数**，因此每个实例可以维护一份被返回对象的独立的拷贝。

## 3.1 组件传值

<h3>父传子
通过props传递，在子组件中注册props属性。

子传父同样可以用此方法，通过props传递父组件的方法给子组件，子组件调用方法，达到子传父的效果

```vue

<template>
	<div>
        <div>
        	父组件    
    	</div>
		<div>
        	需要传递给子组件的值：{{toChild}}    
    	</div>
        <Child :fatherVal="toChild" :upCommit="receiveChildVal" @childTrans="receiveChildVal"></Child>
        <div>
            子组件传过来的值：{{childTOfather}}
    	</div>
        
    </div>
</template>
<script>
    import Child from './components/Child.vue';
    export default{
        data(){
            return{
                toChild:"你好！儿子",
                childTOfather:null
            }
        },
        methods:{
            receiveChildVal:function(cval){
                this.childTOfather=cval
            }
        },
        components: {
    		Child,
  		},
    }
</script>
```

```vue
<template>
	<div>
        <div>
        	子组件    
    	</div>
		<div @click="toFaVal(toFather)">
        	需要传递给父组件的值：{{toFather}}    
    	</div>
        <div @click="toFaVal2(toFather)">
        	子传父法二    
    	</div>
        <div>
            父组件传过来的值：{{fatherVal}}
    	</div>
        
    </div>
</template>
<script>
    export default{
        props:["fatherVal","upCommit"],
        data(){
            return{
                
            }
        },
        methods:{
            toFaVal:function(tof){
                this.$emit("childTrans",tof)
            },
            toFaVal2:function(tof){
                this.upCommit(tof)
            }
        },
    }
</script>
```

<h3>子传父
通过触发当前实例的事件，绑定父组件的事件处理函数，然后将值传给父组件。

**this.$emit( eventName, […args] )**触发当前实例上的事件，this.$emit通常也用作自定义事件。



子传父同样可以用props方法，通过props传递父组件的方法给子组件，子组件调用方法，达到子传父的效果。**这种方法最常用**

<h3>通过实例引用传值

</h3>

**this.$parent，this.$children**，通过引用实例同样也可以达到传值的效果，但推荐不适用，我们应尽量减低组件之间的耦合。

## 3.2 插槽

内容分发

```vue
<template>
	<div>
        <div>插槽——内容分发</div>
       <slot-com>
    		<div>
                 美丽新世界
            </div>
            <div>
                 你好，可以存放多个根节点。
            </div>
    	</slot-com>
    </div>
</template>
<script>
    Vue.component("slot-com",{
        template:`<div>
                    <h3>温馨提示</h3>
                    <slot></slot>
               </div>`
    })
    export default{
        data(){
            return{
                
            }
        },
        methods:{
            
        }
    }
</script>
```

## 3.3 动态组件

```vue
<template>
	<div>
       <div>
           <h2>动态组件</h2>
           <button v-for="item,index in tabs" @click="chooseContent(index+1)">{{item}}</button>
           <component :is="com"></component>
       </div>
    </div>
</template>
<script>
     let com1=Vue.component("home-com",{
         template:"<h1>新闻内容</h1>"
     })
     let com2=Vue.component("news-com",{
         template:`<h1>新闻内容</h1>`
     })
     let com3=Vue.component("list-com",{
         template:`<h1>列表内容</h1>`
     })
     let com4=Vue.component("me-com",{
         template:`<h1>我的内容</h1>`
     })
    export default{
        data(){
            return{
               com:com1 
            }
        },
        methods:{
             chooseContent:function(id){
                 console.log("选择内容",id)
                 console.log(this);
                 this.com=this.$options.components["com"+id]
             }
        }
    }
</script>
```









# 4 路由

Vue Router 是 [Vue.js](http://cn.vuejs.org/) 官方的路由管理器。它和 Vue.js 的核心深度集成，让构建单页面应用变得易如反掌。包含的功能有：

- 嵌套的路由/视图表
- 模块化的、基于组件的路由配置
- 路由参数、查询、通配符
- 基于 Vue.js 过渡系统的视图过渡效果
- 细粒度的导航控制
- 带有自动激活的 CSS class 的链接
- HTML5 历史模式或 hash 模式，在 IE9 中自动降级
- 自定义的滚动条行为

安装路由插件：

```bash
npm install vue-router
```

`this.$router` 和 `router` 使用起来完全一样。

我们可以在任何组件内通过 `this.$router` 访问路由器，也可以通过 `this.$route` 访问当前路由

响应路由参数的变化

## 4.1 配置路由

### javascript

```js
//在src/router/index.js
import Vue from 'vue'
import VueRouter from 'vue-router'

//如果在一个模块化工程中使用它，必须要通过 Vue.use() 明确地安装路由功能
Vue.use(VueRouter)
const routes=[
    {
        //重定向
        path:"/",
        redirect:"/home"
        /*redirect:(to)=>{
            console.log(to)//to与this.$route相同
            if(to.query.go=="news"){
                return {name:"news",params:{id:50}}
            }else{
                return "bignews"
            }
            
        }*/
    },
    
]
// 3. 创建 router 实例，
const router= new VueRouter({
    mode:'history',
    base:process.env.BASE_URL,
    routes
})
export default router

//在src/main.js
import Vue from 'vue'
import App from './App.vue'
import router from './router'

Vue.config.productionTip = false
// 4. 创建和挂载根实例。
// 记得要通过 router 配置参数注入路由，
// 从而让整个应用都有路由功能
new Vue({
  router,
  render: h => h(App),
}).$mount('#app')

```

### html

```html
<div> 
   <h1>路由入口</h1>
    <!-- 使用 router-link 组件来导航. -->
    <!-- 通过传入 `to` 属性指定链接. -->
    <!-- <router-link> 默认会被渲染成一个 `<a>` 标签 -->
   <router-link to="/home">Go to Foo</router-link>
   <h1>路由出口</h1>
    <!-- 路由匹配到的组件将渲染在这里 -->
   <router-view></router-view>
</div>
```

## 4.2 动态路由

我们经常需要把某种模式匹配到的所有路由，全都映射到同个组件。

例如，我们有一个 `User` 组件，对于所有 ID 各不相同的用户，都要使用这个组件来渲染。

```js
    {
        //动态路由
        path:'/news/:id?',//动态路由,?问号表示此id可有可无，并且后面可以跟query参数,url的查询参数
        name:'news',
        props:true,//组件传参，降低耦合度，$route,在页面上通过props拿到id的值
            
        // props: { newsletterPopup: false },  //当 props 是静态的时候有用。
            
        // props: (route) => ({ query: route.query.q }),可以通过路由参数，转换返回的参数
            
        component:()=>import('@/pages/News'),
    },
```

**获取动态路径参数：this.$route.params**

#### url的查询参数

查询字符串（URL参数）是指在URL的末尾加上用于向服务器发送信息的字符串（变量）。将“？”放在URL的末尾，然后再加上“参数＝值”，想加上多个参数的话，使用“&”。以这个形式，可以将想要发送给服务器的数据添加到URL中。

```js
//eg
https://baijiahao.baidu.com/s?id=1619273328999923463&wfr=spider&for=pc
```

**获取查询参数：this.$route.query**

#### 动态路由下的组件复用

提醒一下，当使用路由参数时，例如从 `/user/foo` 导航到 `/user/bar`，**原来的组件实例会被复用**。因为两个路由都渲染同个组件，比起销毁再创建，复用则显得更加高效。**不过，这也意味着组件的生命周期钩子不会再被调用**。

复用组件时，想对路由参数的变化作出响应的话，你可以简单地 watch (监测变化) `$route` 对象：

```js
const User = {
  template: '...',
  watch: {
    $route(to, from) {
      // 对路由变化作出响应...
    }
  },
  //导航守卫
  beforeRouteUpdate (to, from, next) {
    // react to route changes...
    // don't forget to call next()
  }
}
```

也可以通过导航守卫（路由钩子函数）来实现对路由对象的监控

#### 通配符路由

```js
{
  // 会匹配所有路径
  path: '*'
}
{
  // 会匹配以 `/user-` 开头的任意路径
  path: '/user-*'
}
```

当使用一个*通配符*时，`$route.params` 内会自动添加一个名为 `pathMatch` 参数。它包含了 URL 通过*通配符*被匹配的部分。

## 4.3 嵌套路由

children

实际生活中的应用界面，通常由多层嵌套的组件组合而成。同样地，URL 中各段动态路径也按某种结构对应嵌套的各层组件

页面展示子组件的位置由\<router-view>\</router-view>定位

```js
  routes: [
    { 
        path: '/user/:id', 
        component: User,
        children: [
             {
                path: '', component: UserHome  //空路由
             },
            {
              // 当 /user/:id/profile 匹配成功，
              // UserProfile 会被渲染在 User 的 <router-view> 中
              path: 'profile',
              component: UserProfile
              children:[//嵌套多层路由
                ...
               ]
            },
            {
              // 当 /user/:id/posts 匹配成功
              // UserPosts 会被渲染在 User 的 <router-view> 中
              path: 'posts',
              component: UserPosts
            }
        ]
    }
  ]
```

**要注意，以 / 开头的嵌套路径会被当作根路径。 这让你充分的使用嵌套组件而无须设置嵌套的路径。**

`children` 配置就是像 `routes` 配置一样的路由配置数组，所以呢，你可以嵌套多层路由。

## 4.4 命名路由

name

有时候，通过一个名称来标识一个路由显得更方便一些，特别是在链接一个路由，或者是执行一些跳转的时候。

```js
//配置
const router = new VueRouter({
  routes: [
    {
      path: '/user/:userId',
      name: 'user',
      component: User
    }
  ]
})
//函数式导航（编程式导航）
router.push({ name: 'user', params: { userId: 123 }})
```

## 4.5 命名视图

有时候想同时 (同级) 展示多个视图，而不是嵌套展示，例如创建一个布局，有 `sidebar` (侧导航) 和 `main` (主内容) 两个视图，这个时候命名视图就派上用场了。你可以在界面中拥有多个单独命名的视图，而不是只有一个单独的出口。如果 `router-view` 没有设置名字，那么默认为 `default`。

## 4.6 路由组件传参

## 4.7 编程式导航

```js
//<router-link :to="...">
router.push(location, onComplete?, onAbort?)
//<router-link :to="..." replace>            
router.replace(location, onComplete?, onAbort?)
//n为路由前进后退的步数
router.go(n)
```

## 4.8 导航守卫

路由钩子函数。

### 全局守卫

```js
const router = new VueRouter({ ... })

//前置守卫                        
router.beforeEach((to, from, next) => {
    // to: Route: 即将要进入的目标 路由对象

    //from: Route: 当前导航正要离开的路由

    //next: Function: 一定要调用该方法来 resolve 这个钩子
    //next( false | {path:'' , ...})
})
//后置守卫
router.afterEach((to, from) => {
  // ...
})
```

### 路由独享守卫

```js
const router = new VueRouter({
  routes: [
    {
      path: '/foo',
      component: Foo,
      beforeEnter: (to, from, next) => {
        // ...
      }
    }
  ]
})
```



### 组件守卫

```js
const Foo = {
  template: `...`,
  beforeRouteEnter (to, from, next) {
    // 在渲染该组件的对应路由被 confirm 前调用
    // 不！能！获取组件实例 `this`
    // 因为当守卫执行前，组件实例还没被创建
    //不过，你可以通过传一个回调给 next来访问组件实例。在导航被确认的时候执行回调，并且把组件实例作为回调方法的参数。
    next(vm => {
    	//通过 `vm` 访问组件实例
  	})
  },
  beforeRouteUpdate (to, from, next) {
    // 在当前路由改变，但是该组件被复用时调用
    // 举例来说，对于一个带有动态参数的路径 /foo/:id，在 /foo/1 和 /foo/2 之间跳转的时候，
    // 由于会渲染同样的 Foo 组件，因此组件实例会被复用。而这个钩子就会在这个情况下被调用。
    // 可以访问组件实例 `this`
  },
  beforeRouteLeave (to, from, next) {
    // 导航离开该组件的对应路由时调用
    // 可以访问组件实例 `this`
  }
}
```

### 导航解析流程

1. 导航被触发。
2. 在失活的组件里调用 `beforeRouteLeave` 守卫。
3. 调用全局的 `beforeEach` 守卫。
4. 在重用的组件里调用 `beforeRouteUpdate` 守卫 (2.2+)。
5. 在路由配置里调用 `beforeEnter`。
6. 解析异步路由组件。
7. 在被激活的组件里调用 `beforeRouteEnter`。
8. 调用全局的 `beforeResolve` 守卫 (2.5+)。
9. 导航被确认。
10. 调用全局的 `afterEach` 钩子。
11. 触发 DOM 更新。
12. 调用 `beforeRouteEnter` 守卫中传给 `next` 的回调函数，创建好的组件实例会作为回调函数的参数传入。

## 4.9 数据获取

有时候，进入某个路由后，需要从服务器获取数据。例如，在渲染用户信息时，你需要从服务器获取用户的数据。我们可以通过两种方式来实现：

- **导航完成之后获取**：先完成导航，然后在接下来的组件生命周期钩子中获取数据。在数据获取期间显示“加载中”之类的指示。
- **导航完成之前获取**：导航完成前，在路由进入的守卫中获取数据，在数据获取成功后执行导航。

从技术角度讲，两种方式都不错 —— 就看你想要的用户体验是哪种。

### 导航完成后

```js
export default {
  data () {
    return {
      loading: false,
      post: null,
      error: null
    }
  },
  created () {
    // 组件创建完后获取数据，
    // 此时 data 已经被 observed 了
    this.fetchData()
  },
  watch: {
    // 如果路由有变化，会再次执行该方法
    '$route': 'fetchData'
  },
  methods: {
    fetchData () {
      this.error = this.post = null
      this.loading = true
      // replace getPost with your data fetching util / API wrapper
      getPost(this.$route.params.id, (err, post) => {
        this.loading = false
        if (err) {
          this.error = err.toString()
        } else {
          this.post = post
        }
      })
    }
  }
}
```

### 导航完成前

```js
export default {
  data () {
    return {
      post: null,
      error: null
    }
  },
  beforeRouteEnter (to, from, next) {
    getPost(to.params.id, (err, post) => {
      //通过回调可以获取组件实例
      next(vm => vm.setData(err, post))
    })
  },
  // 路由改变前，组件就已经渲染完了
  // 逻辑稍稍不同
  beforeRouteUpdate (to, from, next) {
    this.post = null
    getPost(to.params.id, (err, post) => {
      this.setData(err, post)
      next()
    })
  },
  methods: {
    setData (err, post) {
      if (err) {
        this.error = err.toString()
      } else {
        this.post = post
      }
    }
  }
}
```

## 4.10 滚动行为

```js
const router = new VueRouter({
  routes: [...],
  scrollBehavior (to, from, savedPosition) {
           //第三个参数 savedPosition 当且仅当 popstate 导航 (通过浏览器的 前进/后退 按钮触发) 时才可用。
    // return 期望滚动到哪个的位置
    //支持
           //{ x: number, y: number }
           //{ selector: string, offset? : { x: number, y: number }}(offset 只在 2.6.0+ 支持)
  }
})
```

# 5 **状态管理**

```bash
npm install vuex --save
```

![vuex.png](E:/gitRepository/daily/Vue/legend/vuex.png)

## 5.1 状态配置

```js
import Vue from 'vue'
import Vuex from 'vuex'
import state from './state'//state，getters，mutations，actions等都可以像这样从外面导入
import shopCart from './shopCart'
Vue.use(Vuex)
export default new Vuex.Store({
    //相当于data
    state,
    //store的计算属性,可以供多个组件共用
    getters:{
        reverseMsg:function(state){
            return state.msg.split("").reverse().join("")
        },
        //可以通过让 getter 返回一个函数，来实现给 getter 传参
        //this.$store.getters.mixinMsg(12)
        mixinMsg:(state)=>(id)=>{
            return state.msg+id
        }
    },
    //method，同步方法
    mutations:{
        addCount(state){
            state.count++
        },
        setAge(state,val){
            state.age=val; 
        },
        setWeather(state,val){
            console.log('weather',val)
            state.weather=val
        }
    },
    //异步方法
    actions:{
        queryWeather:function({ commit, state },params){
            console.log('请求天气参数',params)
            //fetch浏览器自带请求方法
            let url=`https://free-api.heweather.net/s6/weather/now?location=chengdu&key=db7f3a13f1ef48168d4045817776ffb2`
            fetch(url,{method:"GET"}).then((res)=>res.json()).then((res)=>{
                console.log('action的天气数据',res)
                commit('setWeather',res)
            })
        },
        
    },
    //模块
    modules:{
        shopCart
    }
})
```

