const fs=require('fs');
fs.writeFile('./file/write01.txt',
'lxx，\n世事一场大梦，人生几度秋凉。\n夜来风叶已鸣廊，看取眉头鬓上。\n酒贱常愁客少,月明多被云妨。\n中秋谁与共孤光，把盏凄然北望。\n',
(err)=>{
    if(err) return console.log(err.message);
    console.log('写入成功')
})
function appendContent(data){
    fs.appendFile('./file/write01.txt',
    data,(err)=>{
        if(err) return console.log(err.message);
        console.log('追加成功')
    })
}
appendContent('你好，lxx!\n');
appendContent('双眸剪秋水，十指剥青葱。\n');
appendContent('白雪凝琼貌，明珠点绛唇。\n');
