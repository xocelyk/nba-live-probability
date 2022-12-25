import pandas as pd
import numpy as np
import datetime
import requests
from bs4 import BeautifulSoup

'''
URLs have the following format:
https://www.sportsbookreview.com/betting-odds/nba-basketball/?date={year}%2F{month}%2F{date}
'''

def get_odds(start_date):
    odds_df = pd.DataFrame()
    today_date = datetime.datetime.today()
    today_date = datetime.datetime(today_date.year, today_date.month, today_date.day)
    start_date = datetime.datetime(start_date.year, start_date.month, start_date.day)
    date = start_date
    odds_df = pd.DataFrame()
    while date <= today_date:
        url = f'https://www.sportsbookreview.com/betting-odds/nba-basketball/?date={date.year}%2F{date.month}%2F{date.day}'
        odds_df = odds_df.append(scrape_odds(url))
        date += datetime.timedelta(days=1)
        break
    return odds_df

def scrape_odds(url):

    # find <div id='tbody-nba'>
    odds_df = pd.DataFrame()
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    tbody = soup.find('div', id='tbody-nba')
    if tbody is None:
        return odds_df
    else:
        # find class="GameRows_eventMarketGridContainer__GuplK GameRows_neverWrap__gnQNO GameRows_compact__ZqqNS bckg-dark"
        # print(tbody.prettify())
        # go thrgouh all the divs
        for div in tbody.find_all('div'):
            print(div.prettify())
        assert 0 == 1
        


if  __name__ == '__main__':
    odds_df = get_odds(datetime.datetime(2022, 12, 23))
    odds_df.to_csv('test.csv')


