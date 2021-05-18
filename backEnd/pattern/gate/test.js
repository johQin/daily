
let str="<span>hah</span><div>你好</div><img/>";
let strn=str.replace(/(\<.*?\>)/g,(target)=>{
    console.log(target)
    return ""
})
console.log(strn);
