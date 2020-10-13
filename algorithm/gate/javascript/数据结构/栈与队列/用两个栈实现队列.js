/**
 * 1.用一个数组用于入队列，用一个数组用于出队列。
 * 2.当入队列的时候，把数据直接压入栈中，充当入队列。
 * 3.因为队列必须满足先入先出，所以需要将入队列中的所有数全部出栈，依次压入出队列中，出时就能实现先入先出的要求
 * 4.当出队列时，如果出队列还有数据那么直接出队列。
 *              如果出队列已经没有数据了，那么就需要将入队列中的所有数弹到出队列中。
 *                                      如果连入队列里也没有数据了，那么返回为空。
 * 
 */
let input=[],output=[];
function push(node){
    input.push(node);
}
function pop(){
    if(output.length>0){
        return output.pop()
    }else{
        while(input.length>0){
            output.push(input.pop())
        }
        if(output.length>0){
            return output.pop()
        }else{
            return null
        }
    }
}