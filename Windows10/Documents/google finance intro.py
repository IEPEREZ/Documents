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
from collections import defaultdict

STOCK_LIST = ['AAPL', 'VRTX', 'AMZN', 'AMD', 'NVS', 'BSTG']

for i in STOCK_LIST:
	with open('my_dict.json', 'w') as f:
		json.dumps(getQuotes(STOCK_LIST), indent=2)

with open('my_dict.json') as f:
	my_dict = json.load(f)

print my_dict
