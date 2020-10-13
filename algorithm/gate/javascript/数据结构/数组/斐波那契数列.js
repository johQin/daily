function Fibonacci(n)
{
    if(n<0){
        return -1
    }
    if(n==0){
        return 0
    }
    if(n==1){
        return 1
    }
    let first=0,second=1,res=null,i=0;
    while(i<n-1){
        res=first+second;
        first=second;
        second=res;
        i++;
    }
    return res
}