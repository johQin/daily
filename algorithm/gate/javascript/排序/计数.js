//计数排序只适用于正整数并且取值范围相差不大的数组排序使用，它的排序的速度是非常可观的。
function count(arr){
    //找出最大值
    let max=Math.max(...arr);
    //初始化计数数组
    let countArr=[];
    for(let i=0;i<=max;i++){
        countArr.push(0);
    }
    //数组元素计数
    arr.map((item)=>{
        countArr[item]+=1;
    })
    //排序
    let index=0;
    for(let i=0;i<countArr.length;i++){
        while(countArr[i]>0){
            arr[index++]=i;
            countArr[i]--;
        }
    }
}
let arr=[10,2,1,86,12,5,5,6,9,4,8];
count(arr);
console.log(arr);