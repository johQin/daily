//基数排序是一种非比较型整数排序算法，其原理是将数据按位数切割成不同的数字，然后按每个位数分别比较。
function radixSort(target){
    let max=target[0];
    target.forEach((item)=>{
        if(item>max){
            max=item;
        }
    })
    let tenBucket=[];
    for(let i=0;i<10;i++){
        tenBucket.push([]);
    }
    let tempBucket=JSON.parse(JSON.stringify(tenBucket));
    let location=0;
    while(true){
        let dd=Math.pow(10,location);
        if(max<dd){
            break;
        }
        target.forEach((item,index) => {
            tenBucket[Math.floor(item/dd)%10].push(item);
        });
        let index=0;
        tenBucket.forEach((item)=>{
            item.forEach(it=>{
                target[index++]=it;
            })
        })
        
        tenBucket=tempBucket;
        location++;
    }
}
let target=[10,2,1,86,12,5,6,9,4,8];
radixSort(target);
console.log(target);