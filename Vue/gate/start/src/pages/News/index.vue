<template>
    <h1>
        新闻页面:新闻id为{{id}}
    </h1>
</template>
<script>
import axios from 'axios'
function parseRoute(arr){
    const tmp=arr.map((item)=>{
        if(item.hasOwnProperty("children")){
            parseRoute(item.children)
        }
        item['parse']=item.component+item.name
        return item
    })
    return tmp;
}
export default {
    name:"News",
    props:['id'],
    data(){
        return{

        }
    },
    beforeMount(){
        // console.log("haha")
        // console.log("新闻页面加载前",this,this.$route.params)
        // axios.get(`https://free-api.heweather.net/s6/weather/now?location=chengdu&key=db7f3a13f1ef48168d4045817776ffb2`).then(function (response) {
        //         console.log("天气",response);
        // })
  const routes=[
    {
        path:'/',
        name:'home',
        component:'@/pages/Home'
    },
    {
        path:'/news/:id?',//动态路由,?问号表示此id可有可无，并且后面可以跟query参数
        name:'news',
        component:'@/pages/News',
        children:[
            {
                component:'pages/bnews',
                name:'bb'
            },
            {
                component:'pages/snews',
                name:'ss'
            },
        ]

    },
]
const route=parseRoute(routes)
console.log('路由解析',route);
  },
}
</script>
<style scoped>

</style>