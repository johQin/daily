
//1.引入express
const express=require('express');
const path=require('path');
const logger=require('./tool/log/logger.js')
const parser=require('./tool/parse/parser.js')

//npm install body-parserf
const bodyParser = require('body-parser')

//npm install art-template
const artTemplate=require('art-template')

//路由独立
const router=require('./router/index.js');

//2.创建服务
const app=express();

//注册中间件
app.use(middleware1);
app.use(middleware2);
//use函数返回的是express的实例，支持方法链
/**
 * app.use(middleware1).use(middleware2)
*/
//通过函数返回函数的方式，创建可配置的中间件。
app.use(configableMiddleware({expandable:true}))
//
app.use(logger)
// app.use(parser)

//解析post请求的请求体body
// parse application/x-www-form-urlencoded
app.use(bodyParser.urlencoded({ extended: false }))
 
// parse application/json
app.use(bodyParser.json())

//ejs设置模板引擎的类型
app.set('engine view','ejs');
//ejs设置模板的位置
app.set('views',path.join(__dirname,'./view'))//第一个参数不要随便乱改

//独立出的路由犹如中间件一样被引入
app.use(router)

//静态资源访问
app.use(express.static(path.join(__dirname,'public')))
app.use('/node_modules',express.static(path.join(__dirname,'node_modules')))
//3.监听请求
app.get('/',(req,res)=>{
    res.send('express server')
})
app.get('/index',(req,res)=>{//第一种get，/index?age=1&name=qin，可以在req.query中获取
    console.log('查询参数：',req.query)
    res.send('home')
})
app.get('/movie/:name/:age',(req,res)=>{//第二种get，/movie/qin/18，可以在req.params中获取
    console.log('路由参数',req.params)
    res.send('movie')
})
app.post('/form',(req,res)=>{
    console.log(req.body)
    res.send('form post')
})
//返回静态页面
app.get('/view',(req,res)=>{
    res.sendFile(path.join(__dirname,'./view/about.html'))
})
//服务器渲染之art-template
app.get('/serrender',(req,res)=>{
    const mockData={name:'qin',age:12}
    const dynHtml=artTemplate(path.join(__dirname,'./view/dynview.html'),mockData)
    res.end(dynHtml)
})
//服务器渲染之ejs
app.get('/ejsrender',(req,res)=>{
    const mockData={name:'qin',age:12}
    res.render('dynHtml.ejs',mockData)
})

app.listen(3000,function(){
    console.log('http://127.0.0.1:3000')
})
function middleware1(req,res,next){
    req.mw1name='first middleware\'s'
    console.log('第一个中间件');
    next();//要想执行下一个中间件，必须调用next()
}
function middleware2(req,res,next){
    console.log('第二个中间件',req.mw1name);//可以看到中间件之间的req是共享的
    next();
}
function configableMiddleware(options){
    return function(req,res,next){
        console.log('可配置的中间件,配置参数',options)
        next()
    }
}
