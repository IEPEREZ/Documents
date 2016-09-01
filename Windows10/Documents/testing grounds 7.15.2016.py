## testing area
import pandas as pd
import itertools

closing_data = pd.read_pickle('googlestocksdfexample_AAPLINDEX_Close.pickle')
headers = closing_data.columns.values

##comparison
header_comb = itertools.combinations(headers, 2)

print header_comb

##for i in header_comb:
	##tickA, tickB = header[header_comb[i]], header_comb[i]
