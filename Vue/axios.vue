<script>
axios
中文：https://www.kancloud.cn/yunye/axios/234845
github：https://github.com/axios/axios
1.安装
npm install axios
2.引入加载
import axios from "axios"
3.请求
	a.get请求：   获取数据
	 created(){//这里做一个网络请求
      this.$axios.get("http://www.wwtliu.com/sxtstu/blueberrypai/getChengpinDetails.php")  
      //这里的url需要指定，标定数据的来源，
      //这里的this.$axios可以和this.$emit做类比
      .then(res=>{console.log(res);
                  this.msg=res.status})  
                  //res用于存放数据  
                  //根据刚刚console.log(res)获得里面的内容，然后再引用，
      .catch(error=>{console.log(error)})
      //简单说一下这里的一个ES6的一个语法点：
      //箭头函数
      //最简形式：参数值=>返回值，，，，一个参数一个返回值
      //其他形式：(参数1，参数2，...)=>{代码块；return语句}
      //         return语句可有可无，但如果要返回对象要在花括号外边加（）
     }
 	b.post请求    发送数据
 			form-data格式：?name=iwen&age=20
 			x-www-form-urlencoded格式:{name:"iwen",age:20}
 		注意：axios接受的post 请求参数 的格式是form-data格式

 		created(){
				this.$axios.post("http://www.wwtliu.com/sxtstu/blueberrypai/login.php", qs.stringify({//请求参数
					user_id:"iwen@qq.com",
					password:"iwen123",
					verification_code:"crfvw"
				}))
				.then(res =>{console.log(res.data)})
				.catch(error=>{console.log(error)})
		}
4.全局的axios的默认值
	这个是在 main.js中
	axios.defaults.baseURL = 'http://www.wwtliu.com';//网址的前导————域名
	axios.defaults.headers.common['Authorization'] = AUTH_TOKEN;//作者认证
	axios.defaults.headers.post['Content-Type'] = 'application/x-www-form-urlencoded';
5.拦截器
	这个是在 main.js中
	在请求或响应被 then 或 catch 处理前拦截它们。
	在获取和发送数据前对数据进行检查或处理数据格式，以便更好的接收或发送
		// 添加请求拦截器
			axios.interceptors.request.use(function (config) {
    	// 在发送请求之前做些什么
    		return config;
  			}, function (error) {
    	// 对请求错误做些什么
    		return Promise.reject(error);
  			});

		// 添加响应拦截器
		axios.interceptors.response.use(function (response) {
    	// 对响应数据做点什么
    		return response;
  			}, function (error) {
    	// 对响应错误做点什么
   			return Promise.reject(error);
 			 });
6.跨域处理
    安装配置文件 cnpm install --save-dev express mysql
    第一步：修改config/index.js文件中的
    proxyTable:{
      "/api":{
        target:"url",//需要跨去的域名
        changeOrigin:true,
        pathRewrite:{
          '^/api':''
        }
      }
    }
    第二步：然后在 main.js中添加host
    Vue.prototype.HOST='/api'//这里HOST代表的是上面target的url值
    第三步：修改 posts.vue中的created函数，那个网址
    var url=this.HOST+域名下的子文件夹
    created(){
      this.$axios.post("url",{........
    第四步：相应的要删除全局的axios的默认值，不然他会在更改 的域名前加上baseURL

    注意：此种跨域解决方案，只能适用于测试阶段。打包的时候，不具备服务器，后端和前端融合在一起了，就不存在跨域问题了

  第一步安装依赖
    cnpm --save-dev mockjs 
    var mock = require("mockjs") 

</script>
<script>
  7.mock数据模拟
  三种常见的方法：
    a.自己创建json文件，使用get请求形式访问数据
        优点：方便，快捷
        缺点：只能存在get请求
    b.项目集成服务器，模拟各种接口（常用）
        优点：模拟真实线上环境
        缺点：增加开发成本
    c.直接使用线上数据
        优点：真实
        缺点：不一定每个项目都存在
    d.数据模拟库mockjs（连接时，需要vpn）
      
    第一步安装依赖
      cnpm --save-dev mockjs 
    第二步 router.js中
    var Mock = require( "mockjs" );
    router.get("/mockjs",
        function(req,res){
            var data = Mock.mock(//mockjs中有很多的数据模板到时候套用就OK了。实在想了解可以参考他的语法规范
                          {'list|1-100':
                            [{'id|+1':1,
                            'name':"iwen"}]
                          }
                        );
            res.json(200,data)
        }
    )
</script>