import requests
import pandas as pd
from plotly import graph_objs as go
from bs4 import BeautifulSoup as bs
import time

def get_boxscore_ids(year):
    ids = {}
    for month in ['october', 'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june']:
        print(month)
        url = 'https://www.basketball-reference.com/leagues/NBA_' + str(year) + '_games-' + month + '.html'
        req = requests.get(url)
        time.sleep(6)
        if req.status_code == 200:
            soup = bs(req.content, 'html.parser')
            table = soup.find('table')
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) > 0:
                    visitor_csk = cells[1].get('csk')
                    visitor_stat_type = cells[1].get('data-stat')
                    assert visitor_stat_type == 'visitor_team_name'
                    visitor_href = cells[1].find('a').get('href')
                    visitor_abbr = visitor_href.split('/')[-2]
                    boxscore_id = visitor_csk.split('.')[1]
                    home_stat_type = cells[3].get('data-stat')
                    assert home_stat_type == 'home_team_name'
                    home_href = cells[3].find('a').get('href')
                    home_abbr = home_href.split('/')[-2]
                    ids[boxscore_id] = {'home_abbr': home_abbr, 'away_abbr': visitor_abbr}
                    print(boxscore_id, home_abbr, visitor_abbr)

    return ids

def get_row_info(row):
    cells = row.find_all('td')
    activity_idxs = [1, 5]
    away_in, away_out, away_event, away_score, home_in, home_out, home_event, home_score = None, None, None, None, None, None, None, None
    active = []
    if len(cells) >= 2:
        if 'enters the game for' in cells[1].text:
            links = cells[1].find_all('a')
            if len(links) > 0:
                # get href
                away_in = links[0].get('href').split('/')[-1][:-5]
                away_out = links[1].get('href').split('/')[-1][:-5] 
    if len(cells) >= 4:
        if 'enters the game for' in cells[5].text:
            links = cells[5].find_all('a')
            if len(links) > 0:
                # get href
                home_in = links[0].get('href').split('/')[-1][:-5]
                home_out = links[1].get('href').split('/')[-1][:-5]
        else:
            if len(cells) >= 2:
                link_list = cells[1].find_all('a')
                for link in link_list:
                    if link.get('href').split('/')[-3] == 'players':
                        active.append(link.get('href').split('/')[-1][:-5])
            if len(cells) >= 5:
                link_list = cells[5].find_all('a')
                for link in link_list:
                    if link.get('href').split('/')[-3] == 'players':
                        active.append(link.get('href').split('/')[-1][:-5])
        away_event = cells[1].text.strip()
        home_event = cells[5].text.strip()
        away_score, home_score = [int(s) for s in cells[3].text.split('-')]
    if len(cells) == 2:
        away_event = cells[1].text
        home_event = cells[1].text     

    return {'AwayScore': away_score, 'AwayEvent': away_event, 'HomeScore': home_score, 'HomeEvent': home_event, 'AwayIn': away_in, 'AwayOut': away_out, 'HomeIn': home_in, 'HomeOut': home_out, 'ActivePlayers': active}

def get_pbp_df(bs_id, home_abbr, away_abbr):
    url = 'https://www.basketball-reference.com/boxscores/pbp/' + bs_id + '.html'
    req = requests.get(url)
    time.sleep(6)
    soup = bs(req.content, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')
    pbp_df = pd.DataFrame(columns=['Period', 'Time', 'AwayName', 'AwayScore', 'AwayEvent', 'HomeName', 'HomeScore', 'HomeEvent', 'AwayIn', 'AwayOut', 'HomeIn', 'HomeOut', 'ActivePlayers'])
    period = 1
    for row in rows:
        cells = row.find_all('td')
        if len(cells) > 0:
            if cells[1].text.startswith('Start of'): #change of quarter
                period += 1
            else:
                if cells[0].text[:1].isnumeric():
                    timestamp = cells[0].text
                    res_row = get_row_info(row)
                    res_row['Period'] = period
                    res_row['Time'] = timestamp
                    pbp_df = pbp_df.append(res_row, ignore_index=True)
        else:
            continue
    pbp_df['HomeName'] = home_abbr
    pbp_df['AwayName'] = away_abbr
    pbp_df[['HomeScore', 'AwayScore']] = pbp_df[['HomeScore', 'AwayScore']].fillna(method='ffill')
    pbp_df[['HomeScore', 'AwayScore']] = pbp_df[['HomeScore', 'AwayScore']].fillna(0)
    assert pbp_df['HomeScore'].isna().sum() + pbp_df['AwayScore'].isna().sum() == 0
    pbp_df[['HomeScore', 'AwayScore']] = pbp_df[['HomeScore', 'AwayScore']].astype(int)
    return pbp_df


def main():
    year = 2023
    boxscore_ids = get_boxscore_ids(year)
    dir = '/Users/kylecox/Documents/ws/nba-pbp/pbp_in_out/'
    for bs_id in boxscore_ids.keys():
        home_abbr = boxscore_ids[bs_id]['home_abbr']
        away_abbr = boxscore_ids[bs_id]['away_abbr']
        print(bs_id)
        pbp_df = get_pbp_df(bs_id, home_abbr, away_abbr)
        print(pbp_df.head(2))
        filename = dir + bs_id + '.csv'
        print(filename)
        pbp_df.to_csv(filename, index=False)


if __name__ == '__main__':
    main()