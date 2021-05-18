<template>
    <div>
        <h1>
            共享数据接收方
        </h1>
       <h3>{{$store.state.count}}</h3>
       <h3>{{count}}</h3>
       <input v-model="age" type='num'/>
       <div>computed计算属性与v-model{{age}}</div>
       <div>mapState的用法：&emsp;{{sexage}}&emsp; {{countXage}}</div>
       <div>{{msgre}}</div>
       <div>
           action的功能测试
           <button @click="askWeather">查询天气</button>
           {{weather==null?'无':weather.HeWeather6[0].now.cond_txt}}
       </div>
    </div>  
</template>
<script>
import { mapState,mapGetters,mapMutations,mapActions } from 'vuex'
export default {
    name:"shareReceive",
    data(){
        return{
            sex:"f"
        }
    },
    methods:{
       askWeather:function(){
           this.$store.dispatch('queryWeather',{address:'chengdu',type:'now'})
       }
    },
    computed:{
        count:function(){
            return this.$store.state.count
        },
        weather:function(){
            return this.$store.state.weather
        },
        age:{
            get:function(){
                return this.$store.state.age
            },
            set:function(val){
                this.$store.commit("setAge",val)
            }
        },
       msgre:function(){
           return this.$store.getters.reverseMsg
       },
       //同样可以使用mapGetters


        //...mapState(["count","age"]),//这样也可以映射，

         //mapStat只是从store映射到组件，并不支持从组件到store，如果组件想把值同步到store，那么还是用上面的age的set和get方式稳妥
        ...mapState({
            //account:"account"
            countXage: state => state.count*state.age,
            // 为了能够使用 `this` 获取局部状态，必须使用常规函数
            sexage(state){
                 return  this.sex+state.age
             }
        })

    },
    beforeMount(){
        console.log('receive',this)
    }
}
</script>
<style scoped>

</style>