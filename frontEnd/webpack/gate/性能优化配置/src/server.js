/*
服务器代码
启动服务器：node server.js
访问服务器地址：127.0.0.1:3000
*/
const express=requier('express');
const app=express();
app.use(express.static('build',{maxAge:1000*3600}));
app.listen(3000);
