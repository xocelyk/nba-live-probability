import os
import numpy as np
import pandas as pd

def find_event(row):
    if not isinstance(row['HomeEvent'], float):
        return str(row['HomeEvent'])
    elif not isinstance(row['AwayEvent'], float):
        return str(row['AwayEvent'])
    else:
        return ''

def time_string(row):
    minutes = str(row['time_in_period_minutes'])
    seconds = str(row['time_in_period_seconds'])
    if len(seconds) == 1:
        seconds = '0' + seconds
    return minutes + ':' + seconds

def time_remaining(row):
    if row['time_elapsed'] <= 48:
        return 48 - row['time_elapsed']
    else:
        return 5 - ((row['time_elapsed'] - 48) % 5)

def time_elapsed(row):
    num_overtimes = row['Period'] - 5
    if row['Period'] < 5:
        return (row['Period'] - 1) * 12 + 12 - row['time_in_period']
    else:
        return 48 + num_overtimes * 5 + 5 - row['time_in_period']

def add_fts(row):
    try:
        num_fts_remaining = 0
        # this condition is kind of bad lol
        if 'free throw' in row['AwayEvent'].lower() and ' of ' in row['AwayEvent'].lower():
            num_fts = row['AwayEvent'].split(' ')[-1]
            num_fts_taken = row['AwayEvent'].split(' ')[-3]
            num_fts_remaining = int(num_fts) - int(num_fts_taken)
        
        if 'free throw' in row['HomeEvent'].lower() and  'of ' in row['HomeEvent'].lower():
            num_fts = row['HomeEvent'].split(' ')[-1]
            num_fts_taken = row['HomeEvent'].split(' ')[-3]
            num_fts_remaining = int(num_fts) - int(num_fts_taken)
        
        if 'shooting foul' in row['AwayEvent'].lower(): #estimate, could be 1, 2, or 3
            num_fts_remaining = 2
        
        if 'shooting foul' in row['HomeEvent'].lower(): #estimate
            num_fts_remaining = 2
        
        return num_fts_remaining
    except:
        print(row[['AwayEvent', 'HomeEvent']])
        return 0

def add_possessions(pbp_df):
    pbp_df['HomeEvent'].fillna('', inplace=True)
    pbp_df['AwayEvent'].fillna('', inplace=True)
    possession_nums = np.zeros(len(pbp_df))
    cur_poss = 0
    for idx, row in pbp_df.iterrows():

        if idx == len(pbp_df) - 1:
            possession_nums[idx] = cur_poss
            break
        elif 'defensive rebound' in row['AwayEvent'].lower() or 'defensive rebound' in row['HomeEvent'].lower():
            cur_poss += 1
        elif 'turnover' in row['AwayEvent'].lower() or 'turnover' in row['HomeEvent'].lower():
            cur_poss += 1

        elif 'jump ball' in row['AwayEvent'].lower() or 'jump ball' in row['HomeEvent'].lower():
            cur_poss += 1
            #TODO: what if possession stays with team?
        elif 'makes' in row['AwayEvent'].lower():
            if 'free throw' in row['AwayEvent'].lower():
                if row['AwayEvent'][-1] == row['AwayEvent'][-6]:
                    cur_poss += 1
                else:
                    pass
            else:
                cur_poss += 1
        elif 'makes' in row['HomeEvent'].lower():
            if 'free throw' in row['HomeEvent'].lower():
                if row['HomeEvent'][-1] == row['HomeEvent'][-6]:
                    cur_poss += 1
                else:
                    pass
            else:
                cur_poss += 1
        elif 'end of' in row['AwayEvent'].lower() or 'end of' in row['HomeEvent'].lower():
            cur_poss += 1
        else:
            pass # possession stays the same
        possession_nums[idx] = cur_poss
    
    if 'Possession' in pbp_df.columns:
        pbp_df.drop('Possession', axis=1, inplace=True)
    pbp_df.insert(4, 'Possession', possession_nums)
    return pbp_df

def add_possession_by_team(pbp_df):
    # reset index
    pbp_df.reset_index(drop=True, inplace=True)
    try:
        pbp_df.drop(['HomePossession', 'AwayPossession'], axis=1, inplace=True)
    except:
        pass
    pbp_df.loc[pbp_df['Possession'] == 0, 'Possession'] = 1
    home_possession = np.empty(len(pbp_df))
    away_possession = np.empty(len(pbp_df))
    # pbp_df['HomeScore'].astype(int)
    # pbp_df['AwayScore'].astype(int)
    pbp_df.reset_index(drop=True, inplace=True)
    for period in pbp_df['Period'].unique():
        period_df = pbp_df[pbp_df['Period'] == period]
        # find first index where diff in score change is greater than 0
        first_home_score_idx = period_df[period_df['HomeScore'].diff() > 0].index[0]
        first_away_score_idx = period_df[period_df['AwayScore'].diff() > 0].index[0]
        if first_home_score_idx < first_away_score_idx:
            first_score_idx = first_home_score_idx - 1
            home_possession[first_score_idx] = True
            away_possession[first_score_idx] = False
        else:
            first_score_idx = first_away_score_idx - 1
            home_possession[first_score_idx] = False
            away_possession[first_score_idx] = True

        # find possession at index = first_score_idx
        # iterate backwards through period_df
        prev_poss_num = period_df[period_df.index == first_score_idx]['Possession'].values[0]
        for idx in range(first_score_idx - 1, period_df.index.min() - 1, -1):
            if period_df[period_df.index == idx]['Possession'].values[0] == prev_poss_num:
                home_possession[idx] = home_possession[idx + 1]
                away_possession[idx] = away_possession[idx + 1]
            else:
                home_possession[idx] = not home_possession[idx + 1]
                away_possession[idx] = not away_possession[idx + 1]
            prev_poss_num = period_df[period_df.index == idx]['Possession'].values[0]
        # iterate forward through period_df
        prev_poss_num = period_df[period_df.index == first_score_idx]['Possession'].values[0]
        for idx in range(first_score_idx + 1, period_df.index.max() + 1):
            if period_df[period_df.index == idx]['Possession'].values[0] == prev_poss_num:
                home_possession[idx] = home_possession[idx - 1]
                away_possession[idx] = away_possession[idx - 1]
            else:
                home_possession[idx] = not home_possession[idx - 1]
                away_possession[idx] = not away_possession[idx - 1]
            prev_poss_num = period_df[period_df.index == idx]['Possession'].values[0]
    pbp_df.insert(5, 'HomePossession', home_possession)
    pbp_df.insert(6, 'AwayPossession', away_possession)
    return pbp_df

def load_pbp():
    # We want the following features: Period, Time in Period, Home Score, Away Score (for margin), Home Close Spread, Home Win
    pbp_dict = {}
    for subdir in os.listdir('pbp_data/with_odds'):
        # skip hidden subdirectories
        if subdir[0] == '.':
            continue
        season = subdir
        for file in os.listdir('pbp_data/with_odds/' + subdir):
            if file[0] == '.':
                continue
            game_id = file.split('.')[0]
            print(game_id)
            df = pd.read_csv('pbp_data/with_odds/' + subdir + '/' + file)
            if df.empty:
                continue
            df['boxscore_id'] = game_id
            pbp_dict[game_id] = df
    return pbp_dict

def format(df):
    df['home_margin'] = df['HomeScore'] - df['AwayScore']
    df['home_win'] = int(df.iloc[-1]['home_margin'] > 0)
    df['time_in_period'] = df.apply(lambda row: float(str(row['Time'].split(':')[0])) + float(str(row['Time'].split(':')[1]))/60, axis=1)
    df['time_elapsed'] = df.apply(lambda row: time_elapsed(row), axis=1)
    df['time_remaining'] = df.apply(lambda row: time_remaining(row), axis=1)
    df['time_in_period_minutes'] = (60 * df['time_in_period']) // 60
    df['time_in_period_minutes'] = df['time_in_period_minutes'].astype(int)
    df['time_in_period_seconds'] = (60 * df['time_in_period']) % 60
    df['time_in_period_seconds'] = df['time_in_period_seconds'].astype(int)
    df['string_time_in_period'] = df.apply(time_string, axis=1)
    df['event'] = df.apply(find_event, axis=1)
    try:
        df = add_possessions(df)
        df = add_possession_by_team(df)
    except:
        print('Error adding possessions')
        print(df.head())
        return None
    df['HomePossession'] = df['HomePossession'].astype(int)
    df.rename(columns={'HomeScore': 'home_score', 'AwayScore': 'away_score', 'Period': 'period', 'HomeName': 'home_name', 'AwayName': 'away_name', 'HomePossession': 'home_possession'}, inplace=True)
    # df = df[['period', 'string_time_in_period', 'time_elapsed', 'time_remaining', 'time_in_period', 'home_name', 'away_name', 'home_score', 'away_score', 'home_possession', 'home_margin', 'home_close_spread','home_win', 'HomeEvent', 'AwayEvent']]
    df['fts_remaining'] = df.apply(add_fts, axis=1)

    df['foul'] = df.apply(lambda row: 'foul' in row['AwayEvent'].lower() or 'foul' in row['HomeEvent'].lower(), axis=1)
    df['turnover'] = df.apply(lambda row: 'turnover' in row['AwayEvent'].lower() or 'turnover' in row['HomeEvent'].lower(), axis=1)
    df['steal'] = df.apply(lambda row: 'steal' in row['AwayEvent'].lower() or 'steal' in row['HomeEvent'].lower(), axis=1)
    df['block'] = df.apply(lambda row: 'block' in row['AwayEvent'].lower() or 'block' in row['HomeEvent'].lower(), axis=1)
    df['timeout'] = df.apply(lambda row: 'timeout' in row['AwayEvent'].lower() or 'timeout' in row['HomeEvent'].lower(), axis=1)
    df['offensive_foul'] = df.apply(lambda row: 'offensive foul' in row['AwayEvent'].lower() or 'offensive foul' in row['HomeEvent'].lower(), axis=1)
    df['defensive_foul'] = df.apply(lambda row: 'defensive foul' in row['AwayEvent'].lower() or 'defensive foul' in row['HomeEvent'].lower(), axis=1)
    df['offensive_rebound'] = df.apply(lambda row: 'offensive rebound' in row['AwayEvent'].lower() or 'offensive rebound' in row['HomeEvent'].lower(), axis=1)
    df['defensive_rebound'] = df.apply(lambda row: 'defensive rebound' in row['AwayEvent'].lower() or 'defensive rebound' in row['HomeEvent'].lower(), axis=1)
    # df.set_index('boxscore_id', inplace=True)

    return df

def main():
    pbp_dict = load_pbp()
    pbp_dict_format = {}
    for game_id, pbp_df in pbp_dict.items():
        pbp_dict_format[game_id] = format(pbp_df)
    return pbp_dict_format

