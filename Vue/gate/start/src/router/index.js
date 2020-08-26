import Vue from 'vue'
import VueRouter from 'vue-router'
import  Home from '@/pages/Home'
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
    {
        //命名视图
        path:'/home',
        alias:'/highlight',//路由别名，这里和重定向的区别是，别名是不会改变路由的，而重定向是重新定位到新路由上去的（replace）
        name:'home',
        // component:Home,
        components:{
            nav:()=>import('@/pages/Home/Nav.vue'),
            aside:()=>import('@/pages/Home/Aside.vue'),
            default:Home,
        },
    },
    {
        //动态路由
        path:'/news/:id?',//动态路由,?问号表示此id可有可无，并且后面可以跟query参数
        name:'news',
        props:true,//组件传参，降低耦合度，$route,在页面上通过props拿到id的值
        // props: { newsletterPopup: false },  //当 props 是静态的时候有用。
        // props: (route) => ({ query: route.query.q }),可以通过路由参数，转换返回的参数
        component:()=>import('@/pages/News'),
    },
    {
        //嵌套路由
        path:"/bignews",
        //path:"/bignews/:content",//这里还可以接动态路由，比如说网址/bignews/123/textnews,动态路由和嵌套路由可以混用
        component:()=>import('@/pages/News/BigNews'),
        children:[
            {
                path:"textnews",
                component:()=>import('@/pages/News/TextNews')
            },
            {
                path:"vedionews",
                component:()=>import('@/pages/News/VedioNews')
            }
        ]
    },
    {
        path:"/share",
        name:"share",
        component:()=>import('@/pages/Share'),
        children:[
            {
                path:"receive",
                component:()=>import('@/pages/Share/Receive'),
            },
            {
                path:"shop",
                component:()=>import('@/pages/Share/Shop'),
            }
        ]
    }
]
// const routeArr=[
//     {
//         path:'/',
//         name:'home',
//         //component:Home,
//         component:'@/pages/Home'
//     },
//     {
//         path:'/news/:id?',//动态路由,?问号表示此id可有可无，并且后面可以跟query参数
//         name:'news',
//         component:'@/pages/News',
//         //component:()=>import('@/pages/News'),
//     },
// ]
// function parseRoute(arr){
//     const tmp=arr.map((item)=>{
//         if(item.hasOwnProperty("children")){
//             parseRoute(item.children)
//         }
//         import(`${item.component}`).then((res)=>{
//             console.log(res)
//             item.component=res
//         })
//         return item
//     })
//     return tmp;
// }
// const routes=parseRoute(routeArr);
// console.log('路由配置',routes);
const router= new VueRouter({
    mode:'history',
    base:process.env.BASE_URL,
    routes
})
export default router