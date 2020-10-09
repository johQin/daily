/**
 * 
 * @param {Array} target=[]
 * 希尔排序：事先定义一个区间，通过区间对数组分组（就像排队报一二三，报到一的一个组，报到二的一个组，报到三的一个组，这里的三就是区间大小）
 * 在数组上从前至后，移动起始位置，以这个位置找组内元素。
 * 跨区间移值：将组内元素按照插入法的方式排序（在原数组上进行操作的话，就是将区间的两头进行比较，最后保证第一个元素的下标大于零，比较结束）
 * 
 * 然后区间逐渐缩小，直到gap=0
 */
function hill(target){
    let gap=1;
    let len=target.length;
    while(gap<len){//找到一个最大区间
        gap=gap*3+1;
    }
    while(gap>0){
        for(let i=gap;i<len;i++){//在数组上依次移动，起始位置。
            let j=i-gap;
            let tmp=target[i];
            while(j>=0&&target[j]>tmp){//跨区间移值，跨区间移动
                target[j+gap]=target[j];
                j-=gap;
            }
            target[j+gap]=tmp;//插入
        }
        gap=Math.floor(gap/3);//区间逐渐缩小，直到等于1时，就是一个插入排序。
    }
    return target
}
let res=hill([10,2,1,5,6]);
console.log(res);