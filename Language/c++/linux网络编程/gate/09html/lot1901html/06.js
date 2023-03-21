
var stop_flag=0;
function timeout() {

	var time = new Date();
	var h = time.getHours();
	var m = time.getMinutes();
	var s = time.getSeconds();
	document.getElementById("time_text").value = h+":"+m+":"+s;
	stop_flag = setTimeout("timeout()",1000); 
	
}
function start() {
	timeout();
}

function stop() {
	clearTimeout(stop_flag);
}