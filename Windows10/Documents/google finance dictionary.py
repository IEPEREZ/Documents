from googlefinance import getQuotes
import json
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import re
import pandas_datareader.data as web
import statistics
import matplotlib.pyplot as plt
import csv
import itertools
import simplejson
from collections import defaultdict
import ast
import urllib2
import lxml.html as lh
from bs4 import BeautifulSoup

""""
url = 'http://www.barchart.com/stocks/sectors/-TOP'
soup = BeautifulSoup(open(url))

"""

STOCK_LIST = ['AAPL', 'VRTX', 'AMZN', 'AMD', 'NVS', 'BSTG', 'AAL', 'ADBE', 'ADI', 'ADSK', 'ALXN', 'AMAT', 'AMGN', 'ATVI', 'BBBY', 'BMRN', 'CELG', 'CERN', 'CHKP', 'CHTR', 'COST', 'CSX', 'CTRP', 'CTSH']
SMART_LIST = []


stockdata = json.dumps(getQuotes(STOCK_LIST), sort_keys=True, indent=2, separators=(',', ': '))
stockdata_list_of_dict = ast.literal_eval(stockdata)
LastTradePrice_list = []
print len(STOCK_LIST)

for item in stockdata_list_of_dict:
		 LastTradePrice_list.append(float(item.values()[3]))

##print LastTradePrice_list		 	

def Portfolio_Average(LastTradePrice_list):
	port_sum = 0.0000
	port_avg = 0.0000
	for i in LastTradePrice_list:
		port_sum += i
	port_avg = port_sum/len(LastTradePrice_list)
	return port_avg

print Portfolio_Average(LastTradePrice_list)

df = pd.DataFrame(LastTradePrice_list)

print df