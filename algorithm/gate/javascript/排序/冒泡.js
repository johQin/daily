/**
 * @method 
 * @param {Array} target=[] 
 * @param {string} sort ="asc"
 * @returns {Array} target 排序后的数组
 * @desc 冒泡法：比较数组相邻两个元素，遇到较大或较小的数，交换他们的值。
 * 一轮比较完毕，数组的len-i-1存放一轮比较后的最大值或最小值.
 * isSort 用来检测剩余的未检测的数组元素是否，已经排好序，如果没有做任何调整那么，说明剩余的数组元素已经是有序的了
 * 时间复杂度O(n^2)
 */
//
function bubbling( target=[],sort='asc'){
        let len=target.length;
        for(let i=0;i<len;i++){
            let isSort=true;
            for(let j=0;j<len-i-1;j++){
                if(sort=='asc'){
                    if(target[j]>target[j+1]){
                        [target[j],target[j+1]]=[target[j+1],target[j]];
                        isSort=false
                    }
                }else{
                    if(target[j+1]>target[j]){
                        [target[j],target[j+1]]=[target[j+1],target[j]];
                        isSort=false;
                    }
                }

            }
            if(isSort){
                break;
            }
        }
        return target
}
let res=bubbling([10,2,1,5,6],'des')
console.log(res)