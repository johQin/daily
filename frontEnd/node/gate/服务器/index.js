//手写nodejs静态服务器

//1.引入http模块
const http=require('http');
//2.创建服务器
const server=http.createServer();
//3.监听请求
server.on('request',(req,res)=>{
    //返回
    if(req.url==''){
        res.end('hello 3000')
    }else{
        res.end('hello guest')
    }
})
server.listen(3000,function(){
    console.log('node server runing at 127.0.0.1:3000')
})