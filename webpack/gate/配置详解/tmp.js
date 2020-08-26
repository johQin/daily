const {resolve}=require('path')
const HtmlWebpackPlugin=require('html-webpack-plugin')
const TerserWebpackPlugin=require('terser-webpack-plugin')
/**
 * entry:入口起点
 * 取值类型：
 * 1.string：单入口单输出（单chunk单bundle）
 * 2.array：多入口单输出
 * 3.object：多入口多输出,有几个入口，就形成几个chunk，也就输出几个bundle，此时bundle的名字就为对象的key值
 */
module.exports={
    // entry:'src/js/index.js',
    // entry:['src/js/index.js','src/js/add.js'],
    entry:{
        index:['./src/js/index.js','./src/js/add.js'],
        sub:'./src/js/sub.js'
    },
    output:{
        //指定文件名称（[目录]+名称）
        filename:'[name].js',//[name]默认值为main，
        //打包输出的公共路径（目录）
        path:resolve(__dirname,'build'),
        //资源引入的公共路径，比如服务器地址等等，
        //publicPath:'ip/',
        //非入口chunk的名称，这是在split code中optimization和import()形成的非chunk文件，命名
        //chunkFilename:'js/[name]_chunk.js',

        
        //整个库向外暴露的变量名，一般用于dll打包
        //library:[name],
        //变量名添加到哪个上
        //libraryTarget:'',
        //libraryTarget:'window',//变量添加到哪个browser上。
        //libraryTarget:'global',//变量名添加到哪个node上
        //
    },
    module:{
        rules:[
            //loader的配置
            {
                test:/\.css$/,
                //单个loader用loader，多个loader用use
                use:['style-loader','css-loader']
            },
            {
                test:/\.js$/,
                //排除文件夹
                exclude:/node_modules/,
                //只检查
                include:resolve(__dirname,'src'),
                enforce:'pre',//or post，pre为优先执行，post延后执行，默认mid中间执行，
                loader:'eslint-loader',
                options:{
                    //自定义配置
                }
            },
            // {
            //     //数组中的loader只生效一个
            //     oneOf:[

            //     ]
            // }
        ]
    },
    plugins:[
        new HtmlWebpackPlugin({
            template:'./src/index.html'
        })
    ],
    mode:'development',
    //解析模块的规则
    resolve:{
        //配置解析模块路径别名,
        //优点：简写路径，缺点：写路径的时候没有提示
        alias:{
            $css:resolve(__dirname,'src/css')
        },
        //配置省略文件路径的后缀名，在引入模块的时候，可以省略文件的后缀名。
        //解析器会按照下面数组的顺序依次匹配，如果没找到相应文件就会报错
        extensions:['js','.json'],
        //告诉webpack解析模块是去找哪个目录，
        //解析器会按照下面数组的顺序依次匹配，如果没找到相应模块就会报错
        modules:[resolve(_dirname,'../node_modules'),'node_modules']
    },
    devServer:{
        //运行代码目录
        contentBase:resolve(__dirname,'build'),
        //监视contentBase目录下的所有文件，一旦文件变化就会reload，疑问：这里和hot有什么区别
        watchContentBase:true,
        watchOptions:{
            //忽略监听某些文件
            ignored:/node_modules/
        },

        //开启gzip压缩
        compress:true,

        port:5000,
        host:'localhost',
        //自动打开浏览器
        open:true,
        //开启HMR
        hot:true,


        //不要显示启动服务器日志信息
        clientLogLevel:'none',
        //除了一些基本启动信息以外，其他内容都不要显示
        quiet:true,
        //如果出错了，不要全屏提示
        overlay:false,

        //服务器代理-->解决开发环境跨域问题
        proxy:{
            //一旦devServer(5000)服务器接收到/api/xxx的请求，就会把请求转发到另外的服务器，
            '/api':{
                target:'http://localhost:3000',
                //发送请求时，请求路径重写。例如这里将/api/xxx-->/xxx(去掉api)
                pathRewrite:{
                    '^api':''
                }
            }
        }
    },
    optimization:{
        splitChunks:{
            
            chunks:'all',
        //下面都是默认值，可以不写
            // //公共规则
            // minSize:30*1024,//分割chunk最小为30kb
            // maxSize:0,//最大没有限制
            // minChunks:1,//要提取的chunk最少被引用一次
            // maxAsyncRequests:5,//按需加载时并行加载的文件的最大数量
            // maxInitialRequests:3,//入口js文件最大并行请求数量

            // name:true,//可以使用命名规则
            // automaticNameDelimiter:'~',//命名链接符为“~”
            // cacheGroups:{
            //     //分割chunk的组
            //     //node_modules文件会被打包到vendors组的chunk中。-->vendors~xxx.js,这里的分割符~是采用上面的分割符
            //     //满足上面的公共规则，如：大小超过30kb，至少被引用一次等等。
            //     vendors:{
            //         test:/[\\/]node_modules[\\/]/,
            //         //优先级
            //         priority:-10
            //     },
            //
            //      //多入口时，被多个chunk引用时
            //     default:{
            //         //要提取的chunk最少被引用两次
            //         minChunks:2,
            //         //优先级
            //         priority:-20,
            //         //如果当前要打包的模块，和之前已经被提取的模块是同一个，就会被复用，而不是重新打包模块
            //         reuseExistingChunk:true,
            //     }

            // },


        },
        //当入口文件（例index.js）通过import()引入了a.js时，并且命名采用了contenthash，打包后main（index）存了此contenthash
        //如果a发生变化，那么因为main存有a的contenthash那么，main文件也要重新打包，main的contenthash值也要变化
        //那么浏览器上的缓存将会失效
        //解决方案：将main文件上记录的hash值用另外一个文件记录起来单独打包，那么main就不会发生变化，
        //通过runtime发挥作用，通过让被引用文件和runtime文件变化，换取mian文件和其他文件的缓存持久化
        runtimeChunk:{
            name:entrypoint=>`runtime-${entrypoint.name}`
        },
        minimizer:[
            //配置生产环境的压缩方案：js和css
            //默认数组第一个元素是针对js的，第二个元素针对css
            new TerserWebpackPlugin({
                //开启缓存
                cache:true,
                //开启多进程打包
                parallel:true,
                //启动source-map
                sourceMap:true,
            })
        ]
            

        
    }
}