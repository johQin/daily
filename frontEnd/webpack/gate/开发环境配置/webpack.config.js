/*
webpack.config.js webpack的配置文件
作用：当你运行webpack指令时，会加载此文件内的配置

所有构建工具都是基于nodejs平台运行的，模块化默认采用Commonjs
*/

//node.js的api接口
//resolve用来拼接绝对路径的方法
const {resolve}=require('path');
const HtmlWebpckPlugin=require('html-webpack-plugin')
//1 创建配置文件webpack.config.js
module.exports={
    //webpack配置
    //1.1.入口起点
    entry:'./src/js/index.js',
    //1.2.输出配置
    output:{
        //__dirname/build/built.js
        //输出文件名
        filename:'js/built.js',
        //输出路径
        //__dirname nodejs的变量，代表当前文件的目录绝对路径
        path:resolve(__dirname,'build')//这里用相对路径也可以'./build/built.js'
        //输出到__dirname/build文件夹下的built.js文件。
    },
    //1.3.loader的配置
    module:{
        rules:[
            //详细loader配置
            //不同文件必须配置不同的loader处理
            {
                //匹配哪些文件,test的值为需要匹配的文件的正则表达式
                //样式文件不会单独打包输出成独立的css等样式文件，而是与js文件为一体的
                test:/\.css$/,//test用于检测正则表达式中的文件
                //匹配到文件后，用哪些loader处理
                //如果使用多个loader，那么使用数组
                use:[
                    //use数组中loader执行顺序：从数组末尾到数组开头，依次执行
                    //创建style标签，将js中的样式插入到进行，添加到head中生效
                    'style-loader',
                    //将css文件变成一个commonjs模块加载到js中，里面内容是样式字符串
                    'css-loader'

                ]
            },
            //每一个loader只能处理一种文件，不能复用。不知道可不可以通过js代码来简化书写方式，后面可自行尝试
            {
                test:/\.less$/,
                use:[
                    'style-loader',
                    'css-loader',
                    //将less文件装换为css文件
                    'less-loader'
                ]
            },
            {
                //问题：默认处理不了html中img图片,这里只能处理css中引入的img文件
                //处理图片资源
                test:/\.(png|jpg|gif)$/,
                //如果仅仅使用单个loader可以直接使用loader
                //下载url-loader file-loader（url-loader会依赖file-loader)
                loader:'url-loader',//url-loader通常用es6的语法处理模块
                //该loader通过options去配置
                options:{
                    //当图片大小小于8kb，就会被base64编码处理，我通常会对小图片做如此处理
                    //在本例程当中vue.jpg将会被base64字符串处理
                    //优点：能够减少请求数量（减轻服务器压力）
                    //缺点：图片体积会变大（文件请求速度变慢）
                    limit:8*1024,

                    // 问题：因为url-loader默认使用es6模块化解析，而html-loader引入图片是commonjs
                    // 解析时会出问题：[object Module]
                    // 解决：关闭url-loader的es6模块化，使用commonjs解析
                    esModule: false,
                    // 给图片进行重命名,如果希望图片名字不要太长，可以如下处理
                    // [hash:10]取图片的hash的前10位
                    // [ext]取文件原来扩展名
                    name: '[hash:10].[ext]',
                    //打包输出文件夹，将图片输出到build/imgs文件夹下
                    outputPath:'imgs'
                }
            },
            {
                test:/\.html$/,
                //用作处理html文件中的img图片（*负责引入img，从而能被url-loader进行处理
                loader:'html-loader'
                //html-loader通常用Commonjs的语法处理模块，
                //但是url-loader是通过es6的语法处理模块，所以接下来url-loader不能处理该图片,还要再url-loader中添加配置
            },

            //打包其他资源（除了html/js/css/img资源）
            {
                //通过test去检测，通过exclude去排除正则表达式中的资源
                exclude:/\.(css|html|js|less|png|jpg|gif)$/,
                loader:'file-loader',
                options:{
                    name:'[hash:10].[ext]',
                    outputPath:'others'
                }
            }
        ]
    },
    //1.4.plugins的配置
    plugins:[
        //html-webpack-plugin

        //new HtmlWebpackPlugin() //功能默认会创建一个空的HTML文件，自动引入被打包后输出的所有资源（js/css)
        //在通常情况下，我们需要使用自定义的Html文件
        //那么将会采用如下结构
        new HtmlWebpckPlugin({
            //复制一个我们自定义./src/index.html文件,并自动引入打包输出的所有资源（js/css)
            template:'./src/index.html'
        })

        
    ],
    //1.5.模式
    mode:'development',//development or production

    //1.6.开发服务器devServer：用来自动化（自动编译，自动打开浏览器，自动刷新浏览器）
    //特点：只会在内存中编译打包，不会有任何输出
    //启动devServer指令为：webpack-dev-server
    devServer:{
        //构建后的项目路径
        contentBase:resolve(__dirname,'build'),
        //启动gzip压缩
        compress:true,
        //启动开发服务器的端口号
        port:3000,
        // 自动打开浏览器
        open: true

    }

}
//2.安装依赖
//3.打包
//4.在build中创建index.html查看效果
