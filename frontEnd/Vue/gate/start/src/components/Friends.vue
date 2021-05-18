<template>
    <div>
        好友列表
        <div>
            <div v-for="item ,index in friends" :key="index">
                <img :src="item.img" :style="{width:'80px',height:'80px'}"/>
                <span>{{item.name}}</span>
            </div>
        </div>
        <!-- <img :src="img"/> -->
    </div>
</template>
<script>
// @符为src下
import hah from '@/assets/lixin/face.jpg'
export default {
    name:"Friends",
    data(){
        return{
            img:hah
        }
    },
    props:["friends"],
    beforeMount(){
        console.log("加载前",this)
    },

}
</script>
<style scoped>

</style>