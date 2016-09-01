book = {}
book['tom'] = {
	'name': 'tom',
	'address': '1 red street, NY',
	'phone': 9899898
}
book['bob'] = {
	'name': 'bob',
	'address': '1 green street, NY',
	'phone': 2323232323
}

import json
s = json.dumps((book), indent=2)

with open("C:\\users\ivan\documents\\book.txt","w") as f:
	f.write(s)
