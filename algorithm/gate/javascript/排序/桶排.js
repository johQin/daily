//高考总分 750 分，全国几百万人，我们只需要创建 751 个桶，循环一遍挨个扔进去，排序速度是毫秒级。
function bucket(arr){
    let min=arr[0],max=arr[0];
    //计算数组的最大最小值
    arr.forEach(item => {
        if(item>max){
            max=item
        }else if(item<min){
            min=item
        }
    });
    //计算桶的数量
    let bucketNum=Math.ceil((max-min)/10);
    //生成桶
    let buckets=[];
    for(let i=0;i<bucketNum;i++){
        buckets.push([]);
    }
    //数据入桶
    arr.forEach((item)=>{
        buckets[Math.floor((item-min)/10)].push(item)
    })
    //桶内排序
    buckets.forEach((item,index)=>{
        buckets[index]=item.sort((a,b)=>{return a-b})
    })
    let index=0;
    //放入原数组。
    buckets.forEach((item)=>{
        item.forEach((it)=>{
            arr[index++]=it;
        })
    })
}
let target=[10,2,1,86,12,5,6,9,4,8];
bucket(target);
console.log(target);