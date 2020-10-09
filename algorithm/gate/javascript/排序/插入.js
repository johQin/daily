/**
 * 
 * @param {Array} target=[] 原数组 
 * @param {string} sort='asc' 排序方式
 * @returns {Array} target 排序后的数组
 * @desc 插入法：如同打扑克摸牌，一边摸一边整理手上牌的顺序，手上的牌是有序的，将摸到的牌插入有序的牌中。
 * 由于是数组，分成三步，
 * 1.找到插入的位置
 * 2.移动数组
 * 3.插入
 */
function insert(target=[],sort='asc'){
    let len=target.length,val=0;
    for(let i=1;i<len;i++){//模拟摸牌所以从第二张牌开始
        val=target[i];
        let j=0;
        for(j=i-1;j>=0;j--){//将摸到的牌与手上的有序数组比较，
            if(val<target[j]){//找到较大或较小的值，向后移动一个
                target[j+1]=target[j];
            }else{//因为手上牌有序，所以一旦不再比当前位置的数较大或较小，那么不再移位，就找到位置
                break;
            }
        }
        target[j+1]=val;//插入数
    }
    return target;
}
let res=insert([10,2,1,5,6])
console.log(res);
