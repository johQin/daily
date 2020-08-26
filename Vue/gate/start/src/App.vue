<template>
  <div id="app">
    <div>
      <!-- <ChatBody class="chatbody"/>
      <Friends class="friends" :friends="friends"/> -->

      <!-- 使用 router-link 组件来导航. -->
      <!-- 通过传入 `to` 属性指定链接. -->
      <!-- <router-link> 默认会被渲染成一个 `<a>` 标签 -->
      <!-- <router-link to='/'>Home</router-link>
      <div>&emsp;&emsp;&nbsp;</div>
      <router-link to='/news'>News</router-link> -->
      <!-- 路由出口 -->
      <!-- 路由匹配到的组件将渲染在这里 -->

      <!-- 命名视图 -->
      <!-- <router-view name="nav"></router-view>
      <router-view name="aside"></router-view> -->
      <router-view/>
    </div>
    
  </div>
</template>

<script>
// @符为src下
import HelloWorld from './components/HelloWorld.vue'
import ChatBody from './components/ChatBody.vue';
import Friends from './components/Friends.vue';
// let ha='@/assets/lixin/face.jpg';
// require(ha)//require不能用变量
export default {
  name: 'App',
  components: {
    ChatBody,Friends
  },
  data(){
    return {
      friends:[
        {name:"欣欣",img:require('./assets/lixin/flower.jpg')},
        {name:"欣花",img:require('@/assets/lixin/face.jpg')},
        {name:"铁欣",img:require('./assets/lixin/metro.jpg')},
        {name:"倾欣",img:require('./assets/lixin/mirror.jpg')},
      ],
      
    }
  },
  beforeMount(){
        console.log("加载前",this)
  },
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
#wrapper{
  width:100vw;
  height:60vh;
  display: flex;
  justify-content: center;
}
.chatbody{
  width:30%;
  height:100%;
  border:solid 1px #000;
}
.friends{
  width:20%;
  height:100%;
  border:solid 1px #000;
}
</style>
