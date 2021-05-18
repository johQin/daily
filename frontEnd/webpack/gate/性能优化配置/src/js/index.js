import '../css/index.css';
import '../css/index.less';
// 引入iconfont样式文件
import '../../public/iconfont/iconfont.css';
import $ from 'jquery';
import print, { mul } from './print';

//通过import动态导入语法：能将某个文件单独打包
// import(/* webpackChunkName:print */'./print').then((print, { mul })=>{
//     print('import()动态导入的方式会使该文件单独打包，以形成多个chunk')
//     console.log('乘法'+(mul(1,2)))
// })
console.log('index.js被加载了');
console.log('jquery引用', $);
function add(x, y) {
  return x + y;
}
const a = 1;
const b = 2;
console.log(`${a}+${b}=${add(a, b)}`);
const timestamp = new Date().getTime();
console.log(`当前时间戳${timestamp}`);
print(`你好，请打印${timestamp}`);
mul(2, 2);

if (module.hot) {
  module.hot.accept('./print.js', () => {
    // 方法会监听print.js文件的变化，一旦发生变化，其他模块不会重新打包构建
    // 并且会执行这个function
  });
}

/**
 * 注册serviceWorker
 * 处理兼容性问题
 * 1.eslint不认识window、navigator全局变量
 * 解决，在package.json中，eslintCOnfig中配置
 * "env":{
 *    "browser":true //支持浏览器端全局变量
 * }
 * 2.sw代码必须运行在服务器上
 *   通过nodejs
 *   通过serve命令，npm i serve -g安装，serve -s build 启动服务器，将build目录下所有资源作为静态资源暴露出去
 * 
 */


if('serviceWorker' in navigator){
  window.addEventListener('load',()=>{
    navigator.serviceWorker.register('/service-worker.js')
    .then(()=>{
      console.log('serviceworker注册成功')
    })
    .catch(()=>{
      console.log('serviceworker注册失败')
    })
  })
}