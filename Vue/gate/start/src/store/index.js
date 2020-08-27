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