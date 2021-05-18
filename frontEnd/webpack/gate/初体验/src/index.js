/*
index.js指定为webpack入口起点文件

1.运行指令：
    a.开发环境：webpack ./src/index.js -o ./build/built.js --mode=development
        指令翻译：webpack会以./src文件/index.js为入口文件开始打包，打包后输出到./build/built.js
            整体打包环境为开发环境
    b.生产环境：webpack ./src/index.js -o ./build/built0.js --mode=production
2.结论
    a.webpack只识别json/js文件，不能处理css/img等文件
    b.生产环境能将es6的模块化（import）编译成浏览器能识别的模块化
    c.生产环境目前比开发环境多一个压缩功能
*/
//引入css文件之后
import './index.css'
/*
ERROR in ./src/index.css 1:9
Module parse failed: Unexpected token (1:9)
You may need an appropriate loader to handle this file type, currently no loaders are configured to process this file. See https://webpack.js.org/concepts#loaders
> html,body{
|     height:400px;
|     width:600px;
 @ ./src/index.js 12:0-20 


 打包文件中已经有报错的代码段了，运行代码的时候会直接报错
 */
import data from './data.json'
console.log('json文件data=',data);
function add(x,y){
    return x+y;
}
let a=1;
let b=2;
console.log(a+"+"+b+"="+add(a,b));