"""
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

web_stats = {
'Day':[1, 2, 3, 4, 5, 6],
'Visitors':[43, 53, 34, 45, 64, 34],
'Bounce_Rate':[65, 72, 62, 64, 54, 66]
}

df = pd.DataFrame(web_stats)

##df.set_index('Day', inplace=True)
##print df.head()

##print df['Visitors']

##print df[['Visitors','Day']]

print df.Visitors.tolist()
print np.array(df[['Bounce_Rate','Visitors']])

df2 = pd.DataFrame(np.array(df[['Bounce_Rate','Visitors']]))

print df2

df = pd.read_csv('C:\\Users\Ivan\Downloads\ZILL-Z84501_MLP.csv')
print df.head()

df.set_index('Date', inplace=True)
df.to_csv('newcsv2.csv')

df = pd.read_csv('C:\\Users\Ivan\Documents\\newcsv2.csv', index_col=0)
print df.head()

df.columns = ['Austin_HPI']
print df.head()

df.to_csv('newcsv3.csv')
df.to_csv('newcsv4.csv', header=False)

df = pd.read_csv('C:\\Users\Ivan\Documents\\newcsv4.csv', names=['Date', 'Austin_HPI'], index_col=0)

print df.head()

df.to_html('example.html')

## VIDEO NO. 4

import quandl
import pandas as pd

api_key = '4mU24zxnsPZxxVtF9-6M'
##df = quandl.get('FMAC/HPI_AK', authtoken=api_key)
##print df.head()

fifty_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
print fifty_states[0][0]

for i in fifty_states[0][0][1:]:
 	print "FMAC/HPI_"+str(i)


## VIDEO No.5 
import pandas as pd

df1 = pd.DataFrame({
	'HPI':[80, 85, 88, 85],
	'Int_rate':[2, 3, 2, 2],
	'US_GDP_Thousands':[50, 55, 65, 55]},
	index = [2001, 2002, 2003, 2004])

df2 = pd.DataFrame({
	'HPI':[80, 85, 88, 85],
	'Int_rate':[2, 3, 2, 2],
	'US_GDP_Thousands':[50, 55, 65, 55]},
	index = [2005, 2006, 2007, 2008])

df3 = pd.DataFrame({
	'HPI':[80, 85, 88, 85],
	'Int_rate':[2, 3, 2, 2],
	'Low_tier_HPI':[50, 52, 50, 53]},
	index = [2001, 2002, 2003, 2004])

s = pd.Series([80, 2, 50], index = ['HPI', 'Int_rate', 'US_GDP_Thousands'])
df4 = df1.append(s, ignore_index=True)
print df4

"""
## video no 6
"""
import pandas as pd

df1 = pd.DataFrame({
	'Year': [2001, 2002, 2003, 2004],
	'Int_rate':[2, 3, 2, 2],
	'US_GDP_Thousands':[50, 55, 65, 55]
	})

df2 = pd.DataFrame({
	'HPI':[80, 85, 88, 85],
	'Int_rate':[2, 3, 2, 2],
	'US_GDP_Thousands':[50, 55, 65, 55]},
	index = [2005, 2006, 2007, 2008])

df3 = pd.DataFrame({
	'Year':[2001, 2003, 2004, 2005],
	'Unemployment':[7, 8, 9, 6],
	'Low_tier_HPI':[50, 52, 50, 53]
	})

##df1.set_index('HPI', inplace=True)
##df3.set_index('HPI', inplace=True)

merged = pd.merge(df1, df3, on= 'Year', how='outer')
merged.set_index('Year', inplace=True)

print merged

## Video No. 7 
## from videono 4??

# import quandl
# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use('fivethirtyeight')

# api_key = '4mU24zxnsPZxxVtF9-6M'

 def state_list():
 	fifty_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
	##print fifty_states[0][0]
	return type(fifty_states[0][0][1:])

# def grab_initial_state_data():
# 	states = state_list()
# 	main_df = pd.DataFrame()

# 	for i in states:
# 	 	query = "FMAC/HPI_"+str(i)
# 	 	df = quandl.get(query, a)thtoken=api_key)
# 	 	df.columns = [str(i)]
# 	 	df[i] = (df[i]- df[i][0])/(df[i][0]) * 100.0

# 	 	if main_df.empty:
# 	 		main_df = df
# 	 	else: 
# 	 		main_df = pd.merge(main_df, df, right_index=True, left_index=True)

# 	print main_df.head()

# 	pickle_out = open('fifty_states3.pickle', 'wb')
# 	pickle.dump(main_df, pickle_out)
# 	pickle_out.close()

# def HPI_Benchmark():
# 	df = quandl.get("FMAC/HPI_USA", authtoken=api_key)
# 	df["Value"] = (df["Value"]- df["Value"][0])/(df["Value"][0]) * 100.0
# 	df.columns = ["United States"]
# 	return df

# #grab_initial_state_data()

# fig = plt.figure()
# ax1 = plt.subplot2grid((2,1), (0,0))
# ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)

# HPI_data = pd.read_pickle('fifty_states3.pickle')
# TX_AK_12corr = pd.rolling_corr(HPI_data['TX'], HPI_data['AK'], 12)

# HPI_data['TX'].plot(ax=ax1, label='TX HPI')
# HPI_data['AK'].plot(ax=ax1, label='AK HPI')

# ax1.legend(loc=4)
# TX_AK_12corr.plot(ax=ax2, label='TX_AK_12corr')
# """
# HPI_data['TX12MA'] = pd.rolling_mean(HPI_data['TX'], 12)
# HPI_data['TX12STD'] = pd.rolling_std(HPI_data['TX'], 12)

# print HPI_data[['TX', 'TX12MA', 'TX12STD']].head()

# HPI_data[['TX','TX12MA']].plot(ax=ax1)
# HPI_data[['TX12STD']].plot(ax=ax2)
# """

# plt.legend(loc=4)
# plt.show()

## VIDEO 11 + 12
# import quandl
# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use('fivethirtyeight')

# bridge_height = {'meters':[10.26, 10.31, 10.27, 10.22, 10.23, 6212.42, 10.28, 10.25, 10.31]}
# df = pd.DataFrame(bridge_height)
# df['std'] = pd.rolling_std(df['meters'],2)
# print df

# df_std = df.describe()['meters']['std']
# print df_std

# df = df[ (df['std'] < df_std) ]
# print df 

# df['meters'].plot()
# plt.show()

### VIDEO 13 
### rip from Video 7-11
"""
import quandl
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

api_key = '4mU24zxnsPZxxVtF9-6M'

def state_list():
	fifty_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
	##print fifty_states[0][0]
	return fifty_states[0][0][1:]

def grab_initial_state_data():
	states = state_list()
	main_df = pd.DataFrame()

	for i in states:
	 	query = "FMAC/HPI_"+str(i)
	 	df = quandl.get(query, authtoken=api_key)
	 	df.columns = [str(i)]
	 	df[i] = (df[i]- df[i][0])/(df[i][0]) * 100.0

	 	if main_df.empty:
	 		main_df = df
	 	else: 
	 		main_df = pd.merge(main_df, df, right_index=True, left_index=True)

	print main_df.head()

	pickle_out = open('fifty_states3.pickle', 'wb')
	pickle.dump(main_df, pickle_out)
	pickle_out.close()

def HPI_Benchmark():
	df = quandl.get("FMAC/HPI_USA", authtoken=api_key)
	df["Value"] = (df["Value"]- df["Value"][0])/(df["Value"][0]) * 100.0
	df.columns = ["United States"]
	return df

def mortgage_30y():
	df = quandl.get("FMAC/MORTG", trim_start= "1975-01-01", authtoken=api_key)
	df["Value"] = (df["Value"]- df["Value"][0])/(df["Value"][0]) * 100.0
	df = df.resample('D')
	df = df.resample('M')
	df.columns = ['M30']
	return df

def sp500_data():
	df = quandl.get("YAHOO/INDEX_GSPC", trim_start="1975-01-01", authtoken=api_key)
	df["Adjusted Close"] = (df["Adjusted Close"] - df["Adjusted Close"][0]) / (df["Adjusted Close"][0]) * 100.0
	df = df.resample('M')
	df.rename(columns = {'Adjusted Close':'sp500'}, inplace=True)
	df = df['sp500']
	return df

def gdp_data():
	df = quandl.get("BCB/4385", trim_start="1975-01-01", authtoken=api_key)
	df["Value"] = (df["Value"]- df["Value"][0])/(df["Value"][0]) * 100.0
	df = df.resample('M')
	df.rename(columns = {'Value':'GDP'}, inplace=True)
	df = df['GDP']
	return df

def us_unemployment():
	df = quandl.get("ECPI/JOB_G", trim_start="1975-01-01", authtoken=api_key)
	df["Unemployment Rate"] = (df["Unemployment Rate"]- df["Unemployment Rate"][0])/(df["Unemployment Rate"][0]) * 100.0
	df = df.resample('1D')
	df = df.resample('M')
	return df


sp500 = sp500_data()
US_GDP = gdp_data()
US_unemployment = us_unemployment()
m30 = mortgage_30y()
HPI_data = pd.read_pickle('fifty_states3.pickle')
HPI_bench = HPI_Benchmark()

HPI = HPI_data.join([HPI_bench, m30, US_unemployment, US_GDP, sp500])
HPI.dropna(inplace=True)
print HPI
print HPI.corr()

HPI.to_pickle("HPI.pickle")
"""
"""
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing, cross_validation
from matplotlib import style
from statistics import mean
style.use('fivethirtyeight')

api_key = '4mU24zxnsPZxxVtF9-6M'

def create_labels(cur_hpi, fut_hpi):
	if fut_hpi > cur_hpi:
		return 1
	else: 
		return 0

def moving_average(values):
	return mean(values)

housing_data = pd.read_pickle('C:\\users\ivan\documents\\HPI.pickle')

housing_data = housing_data.pct_change()

housing_data.replace([1.0, np.inf, -np.inf], np.nan, inplace=True)
housing_data.dropna(inplace=True)

housing_data['US_HPI_future'] = housing_data['United States'].shift(-1)

housing_data['label'] = list(map(create_labels, housing_data['United States'], housing_data['US_HPI_future']))

print housing_data

x = preprocessing.scale(np.array(housing_data.drop(['label', 'US_HPI_future'], 1)))
y = np.array(housing_data['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

print clf.score(x_test, y_test)
print housing_data.drop(['label', 'US_HPI_future'], 1).corr()
"""