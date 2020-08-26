const {resolve}=require("path");
const HtmlWebpackPlugin=require('html-webpack-plugin');
const MiniCssExtractPlugin=require('mini-css-extract-plugin')
const OptimizeCssAssetsWebpackPlugin=require('optimize-css-assets-webpack-plugin')
// 设置nodejs环境变量
process.env.NODE_ENV = 'development';
//设置了node的环境变量为development以后，就可以看到浏览器对开发模式下的css兼容
module.exports={
    entry:"./src/js/index.js",
    output:{
        filename:'js/built.js',
        path:resolve(__dirname,'build')
    },
    module:{
        rules:[
            {
                test:/\.css$/,
                use:[
                    //style-loader用于创建style标签，放入html文件中
                    //'style-loader',
                    //MiniCssExtractPlugin.loader取代style-loader。作用：提取css成单独文件
                    MiniCssExtractPlugin.loader,
                    //css-loader将css整合到js文件中
                    'css-loader',
                    /*
                        css兼容性处理：postcss——>postcss-loader postcss-preset-env
                        postcss-preset-env用于帮助postcss找到package.json中browserlist里面的配置，通过配置加载指定的css兼容性样式
                        package.json中browserlist
                        "browserslist":{
                                //默认是生效生产环境，要想生效development的浏览器css兼容配置，需要设置node的环境变量：process.env.NODE_ENV='development'
                                //开发模式的浏览器要求
                                "development":[
                                "last 1 chrome version",//
                                "last 1 firefox version",
                                "last 1 safari version"
                                ],
                                //生产模式的浏览器要求
                                "production":[
                                ">0.2%",//匹配几乎所有的浏览器
                                "not dead",//不需要兼容已经不再维护的浏览器
                                "not op_mini all"
                                ]
                            }
                        */
                   //使用loader的默认配置，直接写postcss-loader
                   //如果要更改loader的配置，则要通过对象的方式进行loader的自定义配置
                   {
                       loader:'postcss-loader',
                       options:{
                           ident:'postcss',
                           plugins:()=>[
                               //postcss的插件
                               require('postcss-preset-env')()
                           ]
                           
                       }
                   }
                ]
            },
            
            // {
                /*
                语法检查，我们在团队工作的时候，团队开发的风格差不多是一致的，
                eslint-loader，eslint
                注意：语法检查只检查源代码，第三方的库是不用检查的
                设置检查规则：在package.json的eslintConfig中设置,让他继承airbnb-base的语法检查规则
                  "eslintConfig":{
                        "extends":"airbnb-base"
                    }
                推荐使用airbnb规则：https://github.com/airbnb
                npm中搜索eslint-将会看见eslint-config-airbnb（包含react语法）或eslint-config-airbnb-base（基本的代码规范）
                在这里采用eslint-config-airbnb-base（https://www.npmjs.com/package/eslint-config-airbnb-base）
                下载安装依赖：eslint-config-airbnb-base  eslint-plugin-import
                故这里需要下载 4个依赖
                */


            //    test:/\.js$/,
            //    exclude:/node_modules/,//一定要排除node_modules，不然它会检查node_modules
            //    loader:'eslint-loader',
            //    options:{
            //        //如果有airbnb语法错误，自动修复eslint的错误
            //         fix:true,
            //    }
            // },
            
            {
                /*
                js兼容性处理
                下载依赖：babel-loader @babel/core @babel/preset-env
                1.基本js兼容性处理-->@babel/preset-env库，问题：只能转换基本语法，promise等不能转换
                2.全部的js兼容性处理-->@babel/polyfill,在js中手动import '@babel/polyfill';引入后再打包，文件将会变大
                    但我只想解决部分的兼容性问题，不想把全部的兼容性问题引入，这样会使文件变得很大
                3. 需要做兼容性处理的就做兼容，按需加载-->core-js
                 */
                test:/\.js$/,
                exclude:/node_modules/,
                loader:'babel-loader',
                options:{
                    //预设：指示babel做怎么样的兼容性处理
                    //presets:['@babel/preset-env']
                    presets:[
                        [
                            '@babel/preset-env',
                            {
                                //按需解决js兼容性问题
                                useBuiltIns:'usage',
                                //指定core-js的版本
                                corejs:{
                                    version:3
                                },
                                //指定兼容性做到那个版本的浏览器
                                targets:{
                                    chrome:'60',
                                    firefox:'60',
                                    ie:'9',
                                    safari:'10',
                                    edge:'17'
                                }
                            }
                        ]
                    ]
                }
            }
        ]
    },
    plugins:[
        new HtmlWebpackPlugin({
            template:'./src/index.html',
            minify:{
                //折叠空格
                collapseWhitespace:true,
                //移除注释
                removeComments:true
            }
        }),
        new MiniCssExtractPlugin({
            //对输出的css文件重命名
            filename:'css/built.css'
        }),
        //压缩css
        new OptimizeCssAssetsWebpackPlugin()
    ],
    //生产环境下，js会自动做压缩
    //UglifyJsPlugins压缩插件,production模式中内置了此插件
    mode:'development'

}