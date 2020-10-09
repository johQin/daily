/**
 * 
 * @param {Array} target=[] 原数组 
 * @param {string} sort='asc' 排序方式
 * @returns {Array} target 排序后的数组
 * @desc 选择法：比较数组相邻两个元素，遇到较大或较小的数，交换他们数组下标索引。
 * 一轮比较完毕，交换索引len-i-1与index对应元素的值，数组的len-i-1存放一轮比较后的最大值或最小值
 * 时间复杂度O(n^2)
 */
function choose(target=[],sort='asc'){
    let index=0,len=target.length;
    for(let i=0;i<len;i++){
        for(let j=0;j<=len-i-1;j++){
            if(sort=='asc'){
                if(target[j]>target[index]){
                    index=j
                }
            }else{
                if(target[index]>target[j]){
                    index=j
                }
            }
        }
        [target[index],target[len-i-1]]=[target[len-i-1],target[index]];
        index=0;
    }
    return target;
}
let res=choose([10,2,1,5,6],'des');
console.log(res)
