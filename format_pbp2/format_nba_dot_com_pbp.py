'''
This script format_pbp_dfs the NBA.com pbp data so that it can be used for training a win probability model
It does not do the loading of the data, that is done across multiple scripts. Workflow is tracked in workflow.txt
'''

def time_remaining(row):
    if row['time_elapsed'] <= 48:
        return 48 - row['time_elapsed']
    else:
        return 5 - ((row['time_elapsed'] - 48) % 5)

def time_elapsed(row):
    num_overtimes = row['period'] - 5
    if row['period'] < 5:
        return (row['period'] - 1) * 12 + 12 - row['time_in_period']
    else:
        return 48 + num_overtimes * 5 + 5 - row['time_in_period']

def time_string(row):
    minutes = str(row['time_in_period_minutes'])
    seconds = str(row['time_in_period_seconds'])
    if len(seconds) == 1:
        seconds = '0' + seconds
    return minutes + ':' + seconds

def add_fts(row):
    if row['actionType'] == 'foul':
        for qualifier in row['qualifiers']:
            if 'freethrow' in qualifier:
                num_fts_remaining = int(qualifier[0])
                return num_fts_remaining
        return 0
    else:
        if row['actionType'] == 'freethrow':
            [num_fts_taken, num_fts_total] = [int(el) for el in row['subType'].split(' of ')]
            num_fts_remaining = num_fts_total - num_fts_taken
            return num_fts_remaining
        else:
            return 0

def format_pbp_df(df):
    df['home_team_name'] = df['hTeam_city'] + ' ' + df['hTeam_name']
    df['away_team_name'] = df['aTeam_city'] + ' ' + df['aTeam_name']
    df['home_score'] = df['scoreHome'].astype(int)
    df['away_score'] = df['scoreAway'].astype(int)
    df['home_margin'] = df['home_score'] - df['away_score']
    df['home_close_spread'] = df['home_spread']
    df['period'] = df['period'].astype(int)
    df['time_in_period'] = df['timeInPeriod']
    df['time_in_period_minutes'] = (60 * df['time_in_period']) // 60
    df['time_in_period_minutes'] = df['time_in_period_minutes'].astype(int)
    df['time_in_period_seconds'] = (60 * df['time_in_period']) % 60
    df['time_in_period_seconds'] = df['time_in_period_seconds'].astype(int)
    df['string_time_in_period'] = df.apply(time_string, axis=1)
    df['time_elapsed'] = df.apply(time_elapsed, axis=1)
    df['time_remaining'] = df.apply(time_remaining, axis=1)
    df['event'] = df['description']

    if df.iloc[-1]['event'] == 'Game End':
        df.iloc[-1]['home_win_prob'] = 0 if df.iloc[-1]['home_margin'] < 0 else 1
        # save as pickle to nba_dot_com_data/tracked_live_games
        df.to_pickle('tracked_game_' + str(df.iloc[0]['game_id']) + '.pickle')
    
    df['home_possession'] = df.apply(lambda row: row['possession'] == row['home_team_id'], axis=1).astype(int)
    df['fts_remaining'] = df.apply(add_fts, axis=1)
    df['foul'] = df.apply(lambda row: 1 if row['actionType'] == 'foul' else 0, axis=1)
    df['turnover'] = df.apply(lambda row: 1 if row['actionType'] == 'turnover' else 0, axis=1)
    df['steal'] = df.apply(lambda row: 1 if row['actionType'] == 'steal' else 0, axis=1)
    df['block'] = df.apply(lambda row: 1 if row['actionType'] == 'block' else 0, axis=1)
    df['timeout'] = df.apply(lambda row: 1 if row['actionType'] == 'timeout' else 0, axis=1)
    df['offensive_foul'] = df.apply(lambda row: 1 if row['subType'] == 'offensive foul' else 0, axis=1)
    df['defensive_foul'] = df.apply(lambda row: 1 if row['foul'] == 1 and row['offensive_foul'] == 0 else 0, axis=1)
    df['offensive_rebound'] = df.apply(lambda row: 1 if row['actionType'] == 'rebound' and row['subType'] == 'offensive' else 0, axis=1)
    df['defensive_rebound'] = df.apply(lambda row: 1 if row['actionType'] == 'rebound' and row['subType'] == 'defensive' else 0, axis=1)
    if df.iloc[-1]['event'] == 'Game End':
        df.iloc[-1]['home_win'] = 0 if df.iloc[-1]['home_margin'] < 0 else 1
    else:
        df.iloc[-1]['home_win'] = None
    return df

def main():
    pbp_dict = 'nba_dot_com_data/pbp_data_2023.pickle'
    pbp_dict_format_pbp_dfted = {}
    x_features = ['time_remaining', 'home_margin', 'home_possession', 'home_close_spread', 'fts_remaining', 'foul', 'turnover', 'steal', 'block', 'timeout', 'offensive_foul', 'defensive_foul', 'offensive_rebound', 'defensive_rebound']
    for boxscore_id, df in pbp_dict.items():
        df = format_pbp_df(df)
        pbp_dict_format_pbp_dfted[boxscore_id] = df
