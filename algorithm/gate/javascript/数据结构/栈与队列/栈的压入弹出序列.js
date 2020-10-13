/**
 *
 */
function IsPopOrder(pushV, popV)
{
    // write code here
    if(pushV.length!=popV.length){
        return false
    }
    let asist=[],len=pushV.length;
    for(let i=0;i<len;i++){
        asist.push(pushV[i]);
        while(asist.length>0 && asist[asist.length-1]==popV[0]){//尤为注意这里的asist的长度要大于0
            asist.pop();
            popV.shift();
        }
    }
    if(asist.length==0){
        return true
    }else{
        return false
    }
}