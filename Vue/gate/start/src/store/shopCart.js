export default{
    namespaced:true,
    state:{
        productNum:10,
        list:[{name:'护发素',price:10.2},{name:'滴眼液',price:86},]
    },
    getters:{
        costTotal:function(state){
            let tmp=0;
            state.list.map((item)=>{
                tmp+=item.price
            })
            return tmp
        }
    },
    mutations:{
        addpnum:function(state){
            state.productNum++
        }
    },
    actions:{
        changepnum:function(content){
            console.log('content',content)
            const {state,commit}=content;
            setTimeout(()=>{
                commit('addpnum')
            },2000)
        }
    }
}