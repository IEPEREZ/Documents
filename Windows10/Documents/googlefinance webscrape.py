from datetime import datetime
import pandas as pd
import numpy as np
import requests
import re
import pandas_datareader.data as web
import statistics
import matplotlib.pyplot as plt

"""
# Specify Date Range
start = datetime(2000, 1, 1)
end = datetime.today()

# Specify symbol
symbol_list = ['AAPL', 'GOOG', 'MSFT', 'AMD', 'PFE', 'AXP', 'C', 'CC', 'ATI', 'BRKR', 'VAR', 'WAT', 'CBT', 'VRTX', 'NVS', 'BSTG', 'INTC', 'TEVA', 'ABT', 'GSK', 'BMY', 'MYL', 'AMZN', 'SNY', 'LLY', 'AIG']

for symbol in symbol_list:
	aapl_from_google = web.DataReader("%s" % symbol, 'google', start, end)
	aapl_from_yahoo = web.DataReader("%s" % symbol, 'yahoo', start, end)

	aapl_from_google.to_csv('%s_from_google.csv' % symbol)
	aapl_from_yahoo.to_csv('%s_from_yahoo.csv' % symbol)
"""

def get_intraday_data(symbol, interval_seconds=301, num_days=10):
    # Specify URL string based on function inputs.
    url_string = 'http://www.google.com/finance/getprices?q={0}'.format(symbol.upper())
    url_string += "&i={0}&p={1}d&f=d,o,h,l,c,v".format(interval_seconds,num_days)

    # Request the text, and split by each line
    r = requests.get(url_string).text.split()

    # Split each line by a comma, starting at the 8th line
    r = [line.split(',') for line in r[7:]]

    # Save data in Pandas DataFrame
    df = pd.DataFrame(r, columns=['Datetime','Close','High','Low','Open','Volume'])

    # Convert UNIX to Datetime format
    df['Datetime'] = df['Datetime'].apply(lambda x: datetime.fromtimestamp(int(x[1:])))

    return df

AAPL_close = get_intraday_data('AAPL', 301, 1)