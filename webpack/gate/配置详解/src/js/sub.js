console.log('引入sub.js')
function sub(a,b){
    let res=a-b;
    console.log(a+'-'+b+'='+res)
    return res
}
export default sub;