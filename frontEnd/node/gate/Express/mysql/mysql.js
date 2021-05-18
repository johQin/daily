//1.引入mysql
var mysql = require('mysql');
//2.创建mysql连接对象
var connection = mysql.createConnection({
  host     : 'localhost',
  user     : 'root',
  password : '123456',
  database : 'offeryxq'
});
//3.开启连接
connection.connect();
//4.执行sql语句
//查
connection.query('SELECT * from student', function (error, results, fields) {
  if (error) throw error;
  console.log('The solution is: ', results);
});
//增
const data1={
    sname:'lxx',
    ssex:1,
    sage:2,
}
connection.query('INSERT INTO student set ?',data1, function (error, results, fields) {
    if (error) throw error;
    console.log('The solution is: ', results);
  });
//改
const data2={
    sid:1,
    sname:'lxxqin',
    ssex:1,
    sage:2,
}
connection.query('update student set ? where sid=?',[data2,1], function (error, results, fields) {
    if (error) throw error;
    console.log('The solution is: ', results);
});
//删
connection.query('delete from student where sid=?',4, function (error, results, fields) {
    if (error) throw error;
    console.log('The solution is: ', results);
});
//5.关闭连接
connection.end();