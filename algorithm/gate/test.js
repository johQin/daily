let arr=[1,"str",function(){return false},{},[],Infinity,0,NaN,undefined,null]
arr.forEach((item)=>{
    if(item){
        console.log(item+"：true")
    }else{
        console.log(item+"：false")
    }
})