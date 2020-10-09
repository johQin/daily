//元素必须是有序的，如果是无序的则要先进行排序操作。
//有序查找，适于静态查找。
//期望时间复杂度为O(log2n)
function BinarySearch(arr,val){
    let len=arr.length;
    let left=0,right=len-1,mid=0;
    while(left<right){
        mid=Math.floor((right-left)/2)+left;
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
function Recurive(arr,val,left,right){
    if(left>right){
        return -1;
    }
    let mid=Math.floor((right-left)/2)+left;
    if(val<arr[mid]){
       return Recurive(arr,val,left,mid-1);
    }else if(val>arr[mid]){
       return Recurive(arr,val,mid+1,right);
    }else{
       return mid;
    }
}
let arr=[1,5,8,9,10,13,46];
let res1=BinarySearch(arr,9);
let res2=Recurive(arr,9,0,arr.length-1);
console.log(res1,res2);
