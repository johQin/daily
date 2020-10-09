const path=require('path');

console.log(__dirname,__filename);
//__dirname：当前文件所在路径
//__filename：dirname + /当前文件名

//join方法可以将相对路径符去掉，换成绝对路径
let curpath=path.join(__dirname,'./index.js');
console.log(curpath);
let uppath=path.join(__dirname,'../文件系统');
console.log(uppath);