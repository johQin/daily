function mylogin(argument) {

	var user = document.getElementById("user").value;
	var pass = document.getElementById("pass").value;

	if(user=="admin" && pass=="123")
	{
		alert("登入成功");
		window.location.href="京东(JD.COM)-正品低价、品质保障、配送及时、轻松购物！.html";

	}
	else
	{
		alert("登入失败");

	}

}

function login_exit() {
	  document.getElementById("user").value="";
	  document.getElementById("pass").value="";
}