//插值法，是二分法的进阶，
//二分法mid的取值是取中间位置，mid=left+Math.floor((right-left)/2)
//而插入法的mid是自适应的。mid=left+((val-a[left])/(arr[right]-arr[left]))*(right-left)
// 对于表长较大，而关键字分布又比较均匀的查找表来说，插值查找算法的平均性能比折半查找要好的多。
//反之，数组中如果分布非常不均匀，那么插值查找未必是很合适的选择。
//查找成功或者失败的时间复杂度均为O(log2(log2n))
function Insert(arr,val){
    let len=arr.length;
    let left=0,right=len-1,mid=0;
    while(left<right){
        mid=left+Math.floor((val-arr[left])/(arr[right]-arr[left]))*(right-left)
        if(val>arr[mid]){
            left=mid+1;
        }else if(val<arr[mid]){
            right=mid-1;
        }else{
            return mid
        }
    }
    return -1;
}
let arr=[1,5,8,9,10,13,46];
let res1=Insert(arr,9);
console.log(res1);