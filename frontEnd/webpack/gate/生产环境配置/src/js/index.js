import '../css/a.css';
import '../css/b.css'; 
// import '@babel/polyfill';//所有js兼容库
// 这里代码参数间没有空格，在airbnb语法将产生语法错误
// 当在webpack_config.js中eslint-loader配置fix：true时，参数间，参数与运算符号间都会被添加空格

function add(x, y) {
  return x + y;
} 
// 当在一行前，写入eslint-disable-next-line注释，它的下一行代码就不执行检查了，下一行eslint所有规则都失效（下一行不进行eslint检查）
// eslint-disable-next-line
console.log(add(2, 5)); 

// 下面这一行将会在ie8上报错，因为不兼容es6的语法
const sub = (x, y) => x - y; 
// eslint-disable-next-line
console.log(sub(10, 5));
const pro = new Promise((resolve) => {
  setTimeout(() => {
    console.log('延时执行完毕');
    resolve();
  },1000);
});
console.log(pro);
