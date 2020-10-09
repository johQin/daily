//归并排序的核心思想是分治，分而治之，将一个大问题分解成无数的小问题进行处理，处理之后再合并
function mergeSort(target){
    let tmp=[];
    target.map(()=>{
        tmp.push(0);
    })
    sort(target,tmp,0,target.length-1);
}
function sort(target,tmp,start,end){
    if(end<=start){
        return ;
    }
    //迭代切分数组，这里只是找到切分位，没有将数组拆分为零散的数组。
    let mid=start+Math.floor((end-start)/2);
    sort(target,tmp,start,mid);
    sort(target,tmp,mid+1,end);

    merge(target,tmp,start,mid,end);
}
function merge(target,tmp,start,mid,end){
    //复制需要合并的数组
    for (let s = start; s <= end; s++) {
        tmp[s] = target[s];
    }
    let left=start;
    let right=mid+1;
    //左右两个数据分叉，都是有序的，所以将两个有序数组，合并一个数组时，采用了双指针。
    //将两个指针中较小的值放入，目标数组中，
    //如果一个数据分叉走完了，另一个数据分叉还有数据，那么直接将剩余分叉上的数据直接赋值给目标数组。
    for(let i=start;i<=end;i++){
        if(left>mid){
            target[i]=tmp[right++]
        }else if(right>end){
            target[i]=tmp[left++]
        }else if(tmp[left]<tmp[right]){
            target[i]=tmp[left++];
        }else{
            target[i]=tmp[right++];
        }
    }

}
let target=[10,2,1,86,12,5,6,9,4,8];
mergeSort(target);
console.log(target);