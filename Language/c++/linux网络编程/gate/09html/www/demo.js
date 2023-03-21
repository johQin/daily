
function getXMLHttpRequest()
{
	var xmlhttp = null;
	if (window.XMLHttpRequest)//自动检测当前浏览器的版本，如果是IE5.0以上的高版本的浏览器
	{// code for IE7+, Firefox, Chrome, Opera, Safari
		xmlhttp=new XMLHttpRequest();//创建请求对象
	}
	else////如果浏览器是底版本的
	{// code for IE6, IE5
		xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");//创建请求对象
	}
	return xmlhttp;//返回请求对象
}

function getdata() {

	//第一步创建对象
	var xmlhttp = getXMLHttpRequest();
	//设置回调函数
	xmlhttp.onreadystatechange=function()
	{
		if(xmlhttp.readyState==4 && xmlhttp.status==200)
		{
			document.getElementById("lab").innerHTML=xmlhttp.responseText;
		}
	}
	//建立对服务器的请求
	xmlhttp.open("GET", "d.txt", true);
	xmlhttp.setRequestHeader("If-Modified-Since", "0");

	xmlhttp.send(); 
}