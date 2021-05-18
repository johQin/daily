# JSON-JavaScript Object Notation

是一种轻量级的数据交换格式

# JSON语法

1. 支持两种数据结构：a.键值对的数据集合{}，eg：对象，记录结构，字典，哈希表
          b.值的有序列表[]（数组）
2. 支持数据类型的格式书写：
     - 数值，字符串""，布尔值，null，
     - 对象{"键":值,}，数组 [值,]
       ps：键一定是字符串类型，值可以是任意支持的数据类型
3. 通常使用引号''将json数据格式整合为json字符串
4. ES5中定义了全局对象JSON，提供了两个方法用于将原生的JavaScript数据结构和json字符串来回的转换
     - stringify(数据结构变量,数据处理 function(key,value))——将JavaScript数据结构转化为json字符串
     - parse(json变量，数据处理 function(key,value))——将json字符串转化为JavaScript的数据结构
       

```javascript
var box = 
	[
		{
		teacher:'a',
		course:'math',
		period:64,
		toJSON:function(){return this.course;}
		//toJSON项用于将指定的属性或处理的方法，转换为json，
		//这是stringify的首要执行项
		},
		{
		teacher:'b',
		course:'english',
		period:48,
		toJSON:function(){return this.course;}
		},
		{
		teacher:'c',
		course:'physics',
		period:96,
		toJSON:function(){return this.course;}
		}
	]
	var json = JSON.stingify(box,
                             function(key, value){
        							if(key=='teacher'){
                                        return 'Mr.'+ value;
                                    }else{
                                        return value}
								}
                            )
				//这里在转换的同时，对里面的值做了处理
				//or
				JSON.stringify(box,['course','teacher'])
				//这里只取出前两个属性和值来做转换
```

