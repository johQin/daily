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

function send_data(argument) {
	// 创建对象
	var xmlhttp = getXMLHttpRequest();
	//设置回调
	xmlhttp.onreadystatechange=function()
	{
		if(xmlhttp.readyState==4 && xmlhttp.status==200)
		{
			document.getElementById("lab").innerHTML=xmlhttp.responseText;
		}
	}
	//open创建请求报文
	xmlhttp.open("GET", "/cgi-bin/a.cgi?100", true);
	xmlhttp.setRequestHeader("If-Modified-Since", "0");//
	//发送

	xmlhttp.send();
	
}