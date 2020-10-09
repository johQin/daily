/**
 * 分块查找分两步：
 * 1.在索引表中，确定待查记录所在的块，可以顺序查找或折半查找索引表（块间有序）
 * 2.在块内顺序查找（块内无序）
 */
function find(key,arr,indT){
    let keys=Object.keys(indT);
    for(let i=0;i<keys.length;i++){
        //找分块
        if(key<keys[i]){
            console.log('key',arr[i],key)

            //分块对应的区间
            let spanleft=indT[keys[i]],spanright;
            if(i==keys.length-1){
                spanright=arr.length-1;
            }else{
                spanright=indT[keys[i+1]]
            }
            console.log(spanleft,spanright);

            //在区间内找元素
            for(let j=spanleft;j<=spanright;j++){
                if(key==arr[j]){
                    return j;
                }
            }
            break;
        }
    }
    return -1;
    
}
let arr=[24,21,6,11,8,22,32,31,54,72,61,78,88,83]
let indexTable={24:0,54:6,78:9,88:12}//key为块中最大值:val为块的开始索引
//索引0~5之间最大值为24
//索引6~8之间最大值为54
//索引9~11之间最大值为78
//索引12~数组末尾之间最大值为88
console.log(find(31,arr,indexTable));