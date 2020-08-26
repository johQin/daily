const {resolve}=require('path')
const HtmlWebpackPlugin=require('html-webpack-plugin')

module.exports={
    entry:'src/js/index.js',
    output:{
        //指定文件名称（[目录]+名称）
        filename:'[name].js',//[name]默认值为main，
        //打包输出的公共路径（目录）
        path:resolve(__dirname,'build'),
       
    },
    module:{
        rules:[
            //loader的配置
            {
                test:/\.css$/,
                //单个loader用loader，多个loader用use
                use:['style-loader','css-loader']
            },
        ]
    },
    plugins:[
        new HtmlWebpackPlugin({
            template:'./src/index.html'
        })
    ],
    mode:'development',
    resolve:{

    }
}