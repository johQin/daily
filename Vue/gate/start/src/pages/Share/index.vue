<template>
    <div>
        <h1>
            共享数量
        </h1>
        <h3>报道：{{count}}</h3>
        <button @click="ccount(count)">添加</button>
        <router-view></router-view>
    </div>  
</template>
<script>
export default {
    name:"Share",
    data(){
        return{
            count:0,
        }
    },
    methods:{
        ccount:function(p){
            this.count=p+1;

            //请求仓库的同步方法，mutation中的addCount
            this.$store.commit('addCount');
        }
    },
    beforeMount(){
        console.log("share",this)
    }
}
</script>
<style scoped>

</style>