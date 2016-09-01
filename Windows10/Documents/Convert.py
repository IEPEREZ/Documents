## time conversion hackerrank ##
str = '12:23:34AM'
time_lst=list(str)

def convert(time_lst):
	if time_lst[8:] =='AM':
		return time_lst[:8]

	elif time_lst[:2] == '12' and time_lst[time8:] == 'AM':
		var_1 = str(time_lst[:8])
		return var_1.replace(var_1[:2],'00')

	else:
	     hh = int(str(time_lst[0:2]) + 12
	     var2 = ''.join(time_lst[:8])
	     return var2.replace(var_1[:2],str(hh))


print(convert(time_lst))