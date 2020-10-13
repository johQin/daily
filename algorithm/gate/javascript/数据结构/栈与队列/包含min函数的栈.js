/**
 * 1.minStack中的每一个元素都是，stack中对应位置之前元素的最小值。
 */
let minStack=[],stack=[];
function push(node)
{

    let len=minStack.length;
    let topmin;
    if(len==0){
        topmin=node
    }else{
        if(node<minStack[len-1]){
            topmin=node
        }else{
            topmin=minStack[len-1]
        }
    }
    minStack.push(topmin);
    stack.push(node)
    return node
}
function pop()
{
    if(minStack.length>0){
        minStack.pop()
        return stack.pop()
    }else{
        return null
    }
}
function top()
{
    let len=stack.length
    if(len>0){
        return stack[len-1]
    }else{
        return null
    }
}
function min()
{
    let len=minStack.length;
    if(len>0){
        return minStack[minStack.length-1]
    }else{
        return null;
    }
    
}