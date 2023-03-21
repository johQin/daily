function calc_cb(argument) {
	var num1 = document.getElementById("data1").value;
	var num2 = document.getElementById("data2").value;
	if(isNaN(num1) || isNaN(num2))//不全为数字
	{
		alert("输入不全为数字");
		document.getElementById("data1").value= "";
		document.getElementById("data2").value="";

	}
	else
	{
		switch(argument)
		{
			case 1:
				var num = Number(num1) + Number(num2);
				document.getElementById("result").value = num;
				break;
			case 2:
				var num = Number(num1) - Number(num2);
				document.getElementById("result").value = num;
				break;



		}


	}
}