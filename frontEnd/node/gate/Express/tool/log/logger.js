const fs=require('fs');
const path=require('path');
function logger(req,res,next){
    let date=new Date();
    let data=date+'----'+req.url+'----'+req.method+'\n';
    fs.appendFile(path.join(__dirname,'./1.txt'),data,(err)=>{
        if(err) return console.log(err.message);
        console.log('日志写入成功');
        next()
    })
}
module.exports=logger;