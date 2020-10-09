function sequence(arr,val){
    let len=arr.length;
    for(let i=0;i<len;i++){
        if(arr[i]==val){
            return i;
        }
    }
    return -1;
}