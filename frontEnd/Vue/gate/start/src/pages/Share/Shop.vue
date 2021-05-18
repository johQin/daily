<template>
    <div>
        <h1>
           商城用于测试store的模块modules功能
        </h1>
        <h3>商品数量：{{shopCart.productNum}}</h3>
        <h3>总计消费：{{costTotal}}</h3>
        <h3>同步改数量：<button @click='addpnum'>mutation</button></h3>
        <h3>异步改数量：<button @click="changepnum">action</button></h3>
    </div>  
</template>
<script>
import {mapState,mapGetters,mapMutations,mapActions} from 'vuex'
let maps=mapState(['shopCart'])
//let mapg=mapGetters(['costTotal'])//有了namespaced=true，那么他们有了前缀，要通过下面的方式映射
let mapg=mapGetters({costTotal:'shopCart/costTotal'})
// let mapm=mapMutations(['addpnum'])
let mapm=mapMutations({addpnum:'shopCart/addpnum'})
// let mapa=mapActions(['changepnum'])
let mapa=mapActions({changepnum:'shopCart/changepnum'})
export default {
    name:"shop",
    data(){
        return{
           
        }
    },
    computed:{
        ...maps,
        ...mapg,
    },
    methods:{
        ...mapm,
        ...mapa,
    },
    beforeMount(){
        console.log("shop",this)
    }
}
</script>
<style scoped>

</style>