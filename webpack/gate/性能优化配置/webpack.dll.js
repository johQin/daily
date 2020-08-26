/**
 * 此js专门用于打包某个库
 * 使用dll技术，对某些库（第三方，jquery、react、vue等等）进行单独打包
 * 当你运行webpack时，默认查找webpack.config.js配置文件
 * 需求：需要运行webpack.dll.js文件，
 * webpack --config webpack.dll.js
 */
const {resolve}=require('path')
const webpack=require('webpack')
module.exports={
    entry:{
        //最终打包生成的chunk名是键
        //值为数组，需要打包的多个库,
        jquery:['jquery',],
        //react:['react','react-dom','react-router-dom']
    },
    output:{
        filename:'[name].js',
        path:resolve(__dirname,'dll'),
        library:'[name]_[hash]',//打包的库里面向外暴露出去的内容叫什么名字
    },
    plugins:[
        //打包生成一个manifest.json-->提供和jquery映射
        new webpack.DllPlugin({
            name:'[name]_[hash]',//映射库的暴露的内容名称
            path:resolve(__dirname,'dll/manifest.json')//输出文件路径
        })
    ],
    mode:'production'
}