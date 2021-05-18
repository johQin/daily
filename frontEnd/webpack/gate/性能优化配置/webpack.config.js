//当修改了webpack配置，更新的配置要想生效，必须重启服务
const {resolve}=require('path');
const HtmlWebpckPlugin=require('html-webpack-plugin')
//const OptimizeCssAssetsWebpackPlugin=require('optimize-css-assets-webpack-plugin')
//const MiniCssExtractPlugin=require('mini-css-extract-plugin')
const WorkboxWebpackPlugin=require('workbox-webpack-plugin')
const webpack=require('webpack')
const AddAssetHtmlWebpackPlugin=require('add-asset-html-webpack-plugin')
/*
    缓存：
        1. babel缓存，cacheDirectory:true
            功能：让第二次打包更快
        2. 文件资源缓存，服务器上配置cache失效时间。
            问题：在cache的失效的最大时间内，文件是不会更新的，即使服务器上的代码已经更新。
            解决：每次webpack构建时的文件名带上一个唯一的hash值，文件名不同会引起浏览器的更新
        问题：如果重新打包（所有文件名造成了不同），会导致所有缓存失效。（可能我只改变了一个文件），
        所以改变一个文件时，我们希望通过某种命名机制，只让被修改的文件的文件名发生变更。
        而我们的webpack提供了一些hash值，供我们使用
        hash：每次webpack构建时，会生成一个唯一的hash值
        chunkhash：根据chunk生成的hash值（如果来自同一个入口文件，那么他们的chunk hash值相同）
        contenthash：根据文件的内容生成hash值。不同的文件hash值一定不一样。
            功能：让代码上线运行缓存更好使用
        
*/

/*
 * 
 */
/**
 * tree shaking:去除无用代码
 * 前提：
 * 1.必须使用es6模块化，
 * 2.开启production环境
 */
module.exports={
    //单入口
    entry:'./src/js/index.js',
    //多入口单chunk
    //entry:['./src/js/index.js','./src/index.html'],
    output:{
        filename:'js/built.js',
        //filename:'js/built.[hash:10].js',//根据每次打包生成的唯一hash值来命名chunk
        path:resolve(__dirname,'build')
    },

    // 分割代码方式1：
    // //多入口多chunk
    // entry:{
    //     index:'./src/js/index.js',
    //     entry2:'./src/js/entry2.js'
    // },
    // output:{
    //     //name：用来取文件名
    //     filename:'js/[name][contenthash:10].js',//根据每次文件内容是否修改的contenthash来命名chunk
    //     path:resolve(__dirname,'build')
    // },
    module:{
        rules:[
            // {
            //    test:/\.js$/,
            //    exclude:/node_modules/,//一定要排除node_modules，不然它会检查node_modules
            //    loader:'eslint-loader',
            //    enforce:'pre',
            //    options:{
            //        //如果有airbnb语法错误，自动修复eslint的错误
            //         fix:true,
            //    }
            // },
            {
                //oneOf中的loader只会匹配一个，而不会重复检测匹配文件
                //注意：oneOf中所有loader的正则表达式匹配的文件不能相同，或者说正则表达式匹配的文件不能有重合的部分
                oneOf:[
                    {
                        test:/\.css$/,
                        use:[
                            'style-loader',
                            'css-loader'
                        ]
                    },
                    {
                        test:/\.less$/,
                        use:[
                            'style-loader',
                            'css-loader',
                            'less-loader'
                        ]
                    },
                    {
                        test:/\.(png|jpg|gif)$/,
                        loader:'url-loader',
                        options:{
                            limit:8*1024,
                            esModule: false,
                            name: '[hash:10].[ext]',
                            outputPath:'imgs'
                        }
                    },
                    {
                        test:/\.html$/,
                        loader:'html-loader'
                    },
                    {
                        exclude:/\.(css|html|js|less|png|jpg|gif)$/,
                        loader:'file-loader',
                        options:{
                            name:'[hash:10].[ext]',
                            outputPath:'others'
                        }
                    },
                    {
                        test:/\.js$/,
                        exclude:/node_modules/,
                        use:[
                            /**
                             * 开启多线程打包
                             * 只有在工作消耗时间比较长，才需要多进程打包
                             * 因为进程启动大概为600ms，进程之间通信也有时间开销
                             */
                            //'thread-loader',
                            // {
                            //     loader:'thread-loader',
                            //     options:{
                            //         workers:2//进程数2个
                            //     }
                            // },

                            {
                                loader:'babel-loader',
                                options:{
                                    presets:[
                                        [
                                            '@babel/preset-env',
                                            {
                                                useBuiltIns:'usage',
                                                corejs:{
                                                    version:3
                                                },
                                                targets:{
                                                    chrome:'60',
                                                    firefox:'60',
                                                    ie:'9',
                                                    safari:'10',
                                                    edge:'17'
                                                }
                                            }
                                        ]
                                    ],
                                    //1.开启babel缓存
                                    //第二次构建时，会读取之前的缓存
                                    //比如说修改一个js文件，不会导致其他大量的js文件的babel更新
                                    cacheDirectory:true,
                                }
                            }
                        ],

                    }
                ],
            },

        ]
    },

    plugins:[
        // new MiniCssExtractPlugin({
        //     filename:'css/built.[hash:10].css'
        // }),
        new HtmlWebpckPlugin({
            template:'./src/index.html'
        }),

        //pwa
        new WorkboxWebpackPlugin.GenerateSW({
            /**
             * 1.帮助serviceworker快速启动
             * 2.删除旧有的serviceworker
             * 生成一个serviceworker配置文件，
             */
            clientsClaim:true,
            skipWaiting:true
        }),
        //在webpack之前，先执行webpack.dll.js，然后再执行本配置文件的打包
        //告诉webpack哪些库不参与打包，同时使用时的名称也得变，
        new webpack.DllReferencePlugin({
            manifest:resolve(__dirname,'dll/manifest.json')
        }),
        //将某个文件已打包的文件引进来，并在html中自动引入该资源
        new AddAssetHtmlWebpackPlugin({
            filepath:resolve(__dirname,'dll/jquery.js')
        })
    ],
    optimization:{
        splitChunks:{
            //1.可以将node_modules中，单独打包成一个chunk然后输出
            //2.自动分析多入口chunk中，有没有公共的文件，如果有会打包成单独的一个chunk，这样就不会重复打包公共的文件
            chunks:'all'
        }
    },
    mode:'development',
    devServer:{
        contentBase:resolve(__dirname,'build'),
        compress:true,
        port:3000,
        open: true,
        /*
        HMR:Hot Module Replacement 热模块替换/模块热替换
        作用：一个模块发生变化，只会重新打包这一个模块
        样式文件：可以使用HMR功能，因为style-loader内部实现了，所以开发环境需要用到 style-loader
        js文件：默认不能使用HMR功能-->需要修改js代码，添加支持HMR功能的代码。js的HMR 功能只能处理非入口js文件，不能处理入口js
        if(module.hot){
                module.hot.accept('./print.js',function(){
                //方法会监听print.js文件的变化，一旦发生变化，其他模块不会重新打包构建
                //并且会执行这个function
                print()
            })
        }
        html文件：默认不能使用HMR功能，同时会导致问题：html文件不能热更新（解决：修改entry入口，将html写入入口文件）了，在开发的时候只有一个html

        */
       hot:true,
    },
    devtool:'source-map'
}
/*
source-map :一种提供源代码到构建后代码的映射技术，如果构建后代码出错了，通过映射可以追踪源代码错误
devtool:[inline-|hidden-|eval-][nosources-][cheap-[module-]]source-map
inline-source-map：内联方式，集中生成一个内联source-map
source-map：外部方式,
hidden-source-map
nosources-source-map
cheap-source-map
cheap-module-source-map
eval-source-map：每一个文件都生成一个对应的source-map,都在eval函数中
1.内部：js文件夹下不会生成.map文件，嵌入在built.js中，用base64的方式压缩并集中。内联方式构建速度更快
2.外部：js文件夹下会生成.map文件，
3.eval：将每一个文件的映射分散在每个js后面，并在eval函数中
*/

