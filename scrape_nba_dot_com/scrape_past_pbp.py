import requests
import json
import os
import pandas as pd
import pickle
import numpy as np
from similar import best_match

# TODO: get game odds and add to dataframe

def get_odds_index_for_schedule(row):
    res = sorted([row['hTeam_odds_name'], row['aTeam_odds_name']])[0] + row['short_date']
    return res

def get_odds_index_for_odds(row):
    res = sorted([row['team1'], row['team2']])[0] + str(row['date'])
    return res

def get_home_open_spread(row):
    #TODO
    odds_index = row['odds_index']
    if odds_index in odds_df.index:
        if odds_df.loc[odds_index]['team1_location'] == 'H':
            return odds_df.loc[odds_index]['team1_open_spread']
        elif odds_df.loc[odds_index]['team1_location'] == 'V':
            return -1 * odds_df.loc[odds_index]['team1_open_spread']
        else:
            raise ValueError('team1_location is not H or A')
    else:
        return None
    
def get_home_close_spread(row):
    odds_index = row['odds_index']
    if odds_index in odds_df.index:
        if odds_df.loc[odds_index]['team1_location'] == 'H':
            return odds_df.loc[odds_index]['team1_close_spread']
        elif odds_df.loc[odds_index]['team1_location'] == 'V':
            return -1 * odds_df.loc[odds_index]['team1_close_spread']
        else:
            print(odds_df.loc[odds_index])
            print(odds_df.loc[odds_index]['team1_location'])
            raise ValueError('team1_location is not H or A')
    else:
        return None

def get_short_date(row):
    date = row['game_date'].split(' ')[0]
    month, day, year = date.split('/')
    if len(str(month)) == 1:
        month = '0' + str(month)
    if len(str(day)) == 1:
        day = '0' + str(day)
    return str(month) + str(day)

def pbp_data(schedule):
    pbp_data = {}
    game_ids = schedule['game_id'].tolist()
    schedule = schedule[schedule['completed'] == True]
    for game_id in game_ids:
        pbp_data[game_id] = {}
        url = 'https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_' + game_id + '.json'
        r = requests.get(url)
        try:
            game_data = r.json()['game']['actions']
        except:
            continue
        game_pbp = {}
        home_open_spread = schedule[schedule['game_id'] == game_id]['home_open_spread'].tolist()[0]
        home_close_spread = schedule[schedule['game_id'] == game_id]['home_close_spread'].tolist()[0]
        home_team_id = schedule[schedule['game_id'] == game_id]['hTeam_id'].tolist()[0]
        away_team_id = schedule[schedule['game_id'] == game_id]['aTeam_id'].tolist()[0]
        home_team_name = schedule[schedule['game_id'] == game_id]['hTeam_name'].tolist()[0]
        away_team_name = schedule[schedule['game_id'] == game_id]['aTeam_name'].tolist()[0]
        home_team_city = schedule[schedule['game_id'] == game_id]['hTeam_city'].tolist()[0]
        away_team_city = schedule[schedule['game_id'] == game_id]['aTeam_city'].tolist()[0]
        for action in game_data:
            actionNumber = action['actionNumber']
            clock = action['clock']
            period = action['period']
            periodType = action['periodType']
            actionType = action['actionType']
            subType = action['subType']
            qualifiers = action['qualifiers']
            personId = action['personId']
            x = action['x']
            y = action['y']
            side = action['side']
            shotDistance = action['shotDistance'] if 'shotDistance' in action else None
            shotResult = action['shotResult'] if 'shotResult' in action else None
            shotActionNumber = action['shotActionNumber'] if 'shotActionNumber' in action else None
            possession = action['possession']
            scoreHome = action['scoreHome']
            scoreAway = action['scoreAway']
            xLegacy = action['xLegacy']
            yLegacy = action['yLegacy']
            isFieldGoal = action['isFieldGoal']
            description = action['description']
            game_pbp[actionNumber] = [clock, period, periodType, actionType, subType, qualifiers, personId, x, y, side, shotDistance, shotResult, possession, scoreHome, scoreAway, xLegacy, yLegacy, isFieldGoal, description, home_open_spread, home_close_spread, home_team_id, away_team_id]
            if description == 'Game End':
                game_pbp = pd.DataFrame.from_dict(game_pbp, orient='index', columns=['clock', 'period', 'periodType', 'actionType', 'subType', 'qualifiers', 'personId', 'x', 'y', 'side', 'shotDistance', 'shotResult', 'possession', 'scoreHome', 'scoreAway', 'xLegacy', 'yLegacy', 'isFieldGoal', 'description', 'home_open_spread', 'home_close_spread', 'home_team_id', 'away_team_id'])
                home_win = scoreHome > scoreAway
                if np.isnan(home_open_spread):
                    print('Error: home_open_spread is nan', game_id)
                    pbp_data.pop(game_id)
                    break
                game_pbp['home_win'] = int(home_win)
                game_pbp['home_open_spread'] = home_open_spread
                game_pbp['home_close_spread'] = home_close_spread
                game_pbp['hTeam_id'] = home_team_id
                game_pbp['aTeam_id'] = away_team_id
                game_pbp['hTeam_possession'] = game_pbp['possession'] == home_team_id
                game_pbp['aTeam_possession'] = game_pbp['possession'] == away_team_id
                game_pbp['timeMinutes'] = game_pbp['clock'].apply(lambda time: float(time[time.find('PT') + 2:time.find('M')]))
                game_pbp['timeSeconds'] = game_pbp['clock'].apply(lambda time: float(time[time.find('M')+1:time.find('S')]))
                game_pbp['timeInPeriod'] = game_pbp['timeMinutes'] + game_pbp['timeSeconds'] / 60
                game_pbp['hTeam_name'] = home_team_name
                game_pbp['aTeam_name'] = away_team_name
                game_pbp['hTeam_city'] = home_team_city
                game_pbp['aTeam_city'] = away_team_city
                pbp_data[game_id] = game_pbp


    return pbp_data


### this should all be its own file
def odds():
    filename = '../odds_data/sportsbookreviews/2023.csv'
    df = pd.read_csv(filename, encoding='latin-1')
    df.replace('NL', np.nan, inplace=True)
    df.dropna(inplace=True)
    df.replace('pk', int(0), inplace=True)
    df.replace('PK', int(0), inplace=True)
    df[['1st', '2nd', '3rd', '4th', 'Final', 'Open', 'Close', 'ML', '2H']] = df[['1st', '2nd', '3rd', '4th', 'Final', 'Open', 'Close', 'ML', '2H']].astype(float)
    df.columns = ['Date', 'Rot', 'VH', 'Team', '1st', '2nd', '3rd', '4th', 'Final', 'Open', 'Close', 'ML', '2H']
    odds_dict = {}
    for i in range(0, len(df), 2):
        team1 = df.iloc[i]
        team2 = df.iloc[i+1]
        team1_location = team1['VH']
        if team1['Open'] > team2['Open']:
            team2_open_spread = team2['Open']
            team1_open_spread = -team2_open_spread
            favored_team = team2['Team']
            total_points_open = team1['Open']
        else:
            team1_open_spread = team1['Open']
            team2_open_spread = -team1_open_spread
            favored_team = team1['Team']
            total_points_open = team2['Open']
        
        if team1['Close'] > team2['Close']:
            team2_close_spread = team2['Close']
            team1_close_spread = -team2_close_spread
            total_points_close = team1['Close']
        else:
            team1_close_spread = team1['Close']
            team2_close_spread = -team1_close_spread
            total_points_close = team2['Close']
        
        team1_score = team1['Final']
        team2_score = team2['Final']
        team1_ml = team1['ML']
        team2_ml = team2['ML']
        team1_margin = team1_score - team2_score
        open_error = team1_margin - team1_open_spread
        close_error = team1_margin - team1_close_spread
        date = team1['Date']
        odds_dict[i] = {'date': date, 'team1_location': team1_location, 'team1': team1['Team'], 'team2': team2['Team'], 'team1_open_spread': team1_open_spread, 'team2_open_spread': team2_open_spread, 'team1_close_spread': team1_close_spread, 'team2_close_spread': team2_close_spread, 'team1_score': team1_score, 'team2_score': team2_score, 'team1_ml': team1_ml, 'team2_ml': team2_ml, 'team1_margin': team1_margin, 'open_error': open_error, 'close_error': close_error, 'favored_team': favored_team, 'total_points_open': total_points_open, 'total_points_close': total_points_close}
    # save odds_dict as pickle
    with open('../nba_dot_com_data/odds_dict_2023.pickle', 'wb') as f:
        pickle.dump(odds_dict, f)
    return odds_dict

odds_dict = odds()
odds_dict_names = []
schedule = pd.read_pickle('../nba_dot_com_data/schedule_2023.pickle')
schedule = schedule[schedule['completed'] == True]

for i in odds_dict:
    if odds_dict[i]['team1'] not in odds_dict_names:
        odds_dict_names.append(odds_dict[i]['team1'])
    if odds_dict[i]['team2'] not in odds_dict_names:
        odds_dict_names.append(odds_dict[i]['team2'])

nba_data_names = []
for game_id, game_data in schedule.iterrows():
    if game_data['hTeam_city'] not in nba_data_names:
        nba_data_names.append(game_data['hTeam_city'] + game_data['hTeam_name'])
    if game_data['aTeam_city'] not in nba_data_names:
        nba_data_names.append(game_data['aTeam_city'] + game_data['aTeam_name'])

odds_names_to_nba_names = {}
for odds_name in odds_dict_names:
    nba_name = best_match(odds_name, nba_data_names)
    odds_names_to_nba_names[odds_name] = nba_name

nba_names_to_odds_names = {}
for odds_name, nba_name in odds_names_to_nba_names.items():
    nba_names_to_odds_names[nba_name] = odds_name

schedule['short_date'] = schedule.apply(get_short_date, axis=1)
schedule['hTeam_odds_name'] = schedule.apply(lambda x: nba_names_to_odds_names[x['hTeam_city'] + x['hTeam_name']], axis=1)
schedule['aTeam_odds_name'] = schedule.apply(lambda x: nba_names_to_odds_names[x['aTeam_city'] + x['aTeam_name']], axis=1)
schedule['odds_index'] = schedule.apply(get_odds_index_for_schedule, axis=1)

odds_df = pd.DataFrame.from_dict(odds_dict, orient='index')
odds_df['odds_index'] = odds_df.apply(get_odds_index_for_odds, axis=1)
odds_df.index = odds_df['odds_index']

schedule['home_open_spread'] = schedule.apply(get_home_open_spread, axis=1)
schedule['home_close_spread'] = schedule.apply(get_home_close_spread, axis=1)
schedule.to_pickle('../nba_dot_com_data/schedule_data_with_odds_2023.pickle')

pbp_data = pbp_data(schedule)
with open('../nba_dot_com_data/pbp_data_2023.pickle', 'wb') as f:
    pickle.dump(pbp_data, f)


