function show_table(argument) {
	var text="<table  border=1>\
	<tr>\
	<th>姓名</th>\
	<th>年龄</th>\
	</tr>\
	<tr>\
	<td>lucy</td>\
	<td>30</td>\
	</tr>\
	<tr>\
	<td>bob</td>\
	<td>20</td>\
	</tr>\
	</table>";

	//document.write(text);
	document.getElementById("d").innerHTML = text;
}