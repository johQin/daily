//快速排序的核心思想是分治法
function quickSort(arr){
    sort(arr,0,arr.length-1);
}

function sort(arr,start,end){
    if(end<=start){
        return;
    }
    let mid=bilateralScanPartition(arr,start,end);
    sort(arr,start,mid-1);
    sort(arr,mid+1,end);
}
//单边扫描法排序
//把目标数组所有值都扫描一遍，以目标数组第一个值为std，以mark作为指针。
//一旦找到比std小的，就与mark指针所指的元素做交换，交换后，mark自加一。
//最后mark所指的位置就是左右序列的分界线（除数组第一个元素std外）
function singleSideScanPartition(arr,start,end){
    let std=arr[start];
    let mark=start;
    for(let i=start+1;i<=end;i++){
        if(arr[i]<std){
            mark+=1;
            [arr[mark],arr[i]]=[arr[i],arr[mark]];
        }
    }
    arr[start]=arr[mark];
    arr[mark]=std;
    return mark;
}

//双边扫描法排序
//left和right两个指针，
//左边序列放比std小的元素，右边序列存放比std大的元素。
//所以一旦从左序列找到一个比std大的，从右边序列找到一个比std小的，交换二者的值。
//因为std取的是start元素的值，所以，要把如果start的值比left元素大，那么就交换两个位置的值。这是为了保证左序列的值小于std。
function bilateralScanPartition(arr,start,end){
    let left=start;
    let right=end;
    let std=arr[start];
    while(left<right){
        while(left<right){
            if(std<arr[left]){
                break;
            }
            left++;
        }
        while(left<right){
            if(arr[right]<std){
                break;
            }
            right--;
        }
        if(left!=right){
            [arr[left],arr[right]]=[arr[right],arr[left]]
        }  
    }
    if(arr[left]<std){
        [arr[left],arr[start]]=[arr[start],arr[left]];
    }
    return left;
}
let arr=[10,2,1,5,6];
quickSort(arr);
console.log(arr);