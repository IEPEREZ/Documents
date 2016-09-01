import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean
from datetime import datetime
from itertools import combinations
import statistics

style.use('fivethirtyeight')

### getting the original dataframe from a set of CSV's in googlefinancewebscrap.py 
def symbol_list():
	symbol_list = ['AAPL', 'GOOG', 'MSFT', 'AMD'] 
 	df = pd.DataFrame(symbol_list)
	return df

def get_Volume():
	main_df = pd.DataFrame()
	symbols = ['AAPL', 'VRTX', 'MSFT', 'AMD']
	return pd.concat([pd.read_csv('%s_from_google.csv' %i) for i in symbols])

print get_Volume()

# def main_df():
#  	symbols = ['AAPL', 'VRTX', 'MSFT', 'AMD']
#  	main_df = pd.DataFrame()

# 	for i in symbols:
# 		df = pd.read_csv('%s_from_google.csv' % (i), delimiter=',', encoding="utf-8-sig")
# 		df.rename(columns = {
# 			'Date':'Date_'+str(i),
# 			'Open':'Open_'+str(i),
# 			'High':'High_'+str(i),
# 			'Low':'Low_'+str(i),
# 			'Close':'Close_'+str(i),
# 			'Volume':'Volume_'+str(i)
# 			}, inplace=True)
# 		if main_df.empty:
# 			main_df = df
# 		else:
# 			main_df = pd.merge(main_df, df, right_index=True, left_index=True)
# 	print main_df.head()
# 	main_df.to_pickle('googlestocksdfexample.pickle')

## main_df() ##praise helix

# google_stocks_example = pd.read_pickle('googlestocksdfexample.pickle')
# google_stocks_example.set_index('Date_AAPL', inplace=True)
# google_stocks_example.to_pickle('googlestocksdfexample_AAPLINDEX.pickle')

# print pd.read_pickle('googlestocksdfexample_AAPLINDEX.pickle').head()


## Quick Separation Example (Apply towards other categories)
"""
def get_close():
	symbols = ['AAPL', 'VRTX', 'MSFT', 'AMD'] ## needs generalization in the form of dynamic pd.series for relevant stocks
 	main_df = pd.DataFrame() ## 
 	source = pd.read_pickle('googlestocksdfexample_AAPLINDEX.pickle')
 	close_df = source[['Close_AAPL', 'Close_VRTX', 'Close_MSFT', 'Close_AMD']]
 	return close_df
 	close_df.to_pickle('googlestocksdfexample_AAPLINDEX_Close.pickle')

close = get_close()
print close.head()
"""
## Data Visualization (needs to explore 3D and staggered representation) 
"""
fig = plt.figure()
ax1 = plt.subplot2grid((2,1), (0,0))
ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)

close_data = pd.read_pickle('googlestocksdfexample_AAPLINDEX_Close.pickle')
AAPL_VRTX_30Dcorr = pd.rolling_corr(close_data['Close_AAPL'], close_data['Close_VRTX'], 30)

close_data['Close_AAPL'].plot(ax=ax1, label='AAPL Closing Prices')
close_data['Close_VRTX'].plot(ax=ax1, label='VRTX Closing Prices')

ax1.legend(loc=4)
AAPL_VRTX_30Dcorr.plot(ax=ax2, label='AAPL_VRTX_30Dcorr')

close_data['AAPL30DMA'] = pd.rolling_mean(close_data['Close_AAPL'], 30)
close_data['AAPL30DSTD'] = pd.rolling_std(close_data['Close_AAPL'], 30)

print close_data[['Close_AAPL', 'AAPL30DMA', 'AAPL30DSTD']].head()

close_data[['AAPL30DMA']].plot(ax=ax1)
close_data[['AAPL30DSTD']].plot(ax=ax2)

plt.legend(loc=4)
plt.show()

## Datavisualization generalization
def Closing_Prices_general():
	close_data = pd.read_pickle('googlestocksdfexample_AAPLINDEX_Close.pickle')
	header = close_data.columns.values
	header_comb = list(itertools.combinations(header, 2))
	for i in header_comb:
		tickA, tickB = header[header_comb[i], header_comb[i]] ## this doesn't make sense 
		tickA_tickB_30Dcorr = pd.rolling_corr(close_data[tickA], close_data[tickB], 30)
		print tickA_tickB_30Dcorr.head()


closepricetest = Closing_Prices_general()
print closepricetest

### second part to be worked on, get rid of str format, write out the appropriate organization forr excel spreadsheets(this needs its own tutorial)
	close_data['Close_%s' %(close_data.ix[TICKA,])]].plot(ax=ax1, label='%s Closing Prices' %(close_data.ix[TICKA]))
	close_data['Close_%s' %(close_data.ix[TICKB,])]].plot(ax=ax1, label='%s Closing Prices' %(close_data.ix[TICKB]))

	ax1.legend(loc=4)
	TICKA_TICKB_30Dcorr.plot(ax=ax2, label='%s_%s_30Dcorr' %())

	close_data['AAPL30DMA'] = pd.rolling_mean(close_data['Close_AAPL'], 30)
	close_data['AAPL30DSTD'] = pd.rolling_std(close_data['Close_AAPL'], 30)

	print close_data[['Close_AAPL', 'AAPL30DMA', 'AAPL30DSTD']].head()

	close_data[['AAPL30DMA']].plot(ax=ax1)
	close_data[['AAPL30DSTD']].plot(ax=ax2)

	plt.legend(loc=4)
	plt.show()
"""