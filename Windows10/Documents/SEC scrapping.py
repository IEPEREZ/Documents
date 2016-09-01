import re

import requests
from bs4 import BeautifulSoup

response = requests.get("https://www.sec.gov/Archives/edgar/data/36405/000093247115006447/indexfunds_final.htm")
soup = BeautifulSoup(response.content)

futures = soup.find_all(text=re.compile('C. Futures Contract'))
for future in futures:
    for row in future.find_next("table").find_all("tr"):
        print [cell.get_text(strip=True) for cell in row.find_all("td")]