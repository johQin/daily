var express = require('express');
var router = express.Router();

//mysql
//1.引入mysql
var mysql = require('mysql');
//2.创建mysql连接对象
var connection = mysql.createConnection({
  host     : 'localhost',
  user     : 'root',
  password : '123456',
  database : 'offeryxq'
});
function sqlexe(sql,params){
    return new Promise((resolve,reject)=>{
        connection.query(sql,params,(err,res)=>{
            console.log(res);
            if(err) throw err;
            resolve(res);
        })
    })

}
//api-curd
router.get('/query/:ssex',(req,res)=>{
    const {query,params} =req;
    sqlexe('select * from student where sage=? and ssex=?',[query.sage,params.ssex]).then((re)=>{
        res.send({
            code:200,
            data:re
        })
    })
})



module.exports=router;