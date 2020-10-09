//斐波那契查找也属于一种有序查找算法。是二分的一个提升。
//随着斐波那契数列的递增，前后两个数的比值会越来越接近0.618
/**
 * https://blog.csdn.net/darkrabbit/article/details/90240507
 * 
 * */ 
function Search(key=0,arr=[]){
    let len=arr.length;
    let fibo=Fibonacci(len);
    let left=0,right=len-1;
    let block=fibo.length-1;//斐波那契数列
    let mid,index;
    while(left<=right){
        mid=left+fibo[block-1]-1;
        index=Math.min(mid,len-1);
        if(key==arr[index]){
            return index;
        }else if(key<arr[index]){
            block-=1;
            right=mid-1;
        }else{
            block-=2;
            left=mid+1;
        }
    }
    return -1;
}
function Fibonacci(targetlen=0){
    let cond=targetlen-1;
    let arr=[];
    arr.push(0,1);
    for(let i=0;arr[i+1]<=cond;i++){//斐波那契数列的最后一个数，要比需要查找数列的length-1要大
        arr.push(arr[i]+arr[i+1]);
    }
    return arr;
}
let arr=[1,1,3,5,6,9,10,55,99];
console.log(Search(99,arr));