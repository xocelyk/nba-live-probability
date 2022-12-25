from live_probability_model import predict
import pickle
import pandas as pd
import os
import numpy as np
from plotly import graph_objects as go
import warnings
# import plotly express
import plotly.express as px


warnings.filterwarnings('ignore')

X_FEATURES = ['time_remaining', 'home_margin', 'home_possession', 'home_close_spread', 'fts_remaining', 'foul', 'turnover', 'steal', 'block', 'timeout', 'offensive_foul', 'defensive_foul', 'offensive_rebound', 'defensive_rebound']

colors = {'ATL': '#E03A3E', 'BRK': '#5F6264', 'BOS': '#007A33', 'CHO': '#00FFFF', 'CHI': '#CE1141', 'CLE': '#860038', 'DAL': '#00538C', 'DEN': '#FDB827', 'DET': '#C8102E',
        'GSW': '#006BB6', 'HOU': '#CE1141', 'IND': '#002D62', 'LAC': '#C8102E', 'LAL': '#552583', 'MEM': '#5D76A9', 'MIA': '#98002E', 'MIL': '#00471B', 'MIN': '#32CD32',
        'NOP': '#0C2340', 'NYK': '#FFA500', 'OKC': '#007AC1', 'ORL': '#0077C0', 'PHI': '#006BB6', 'PHO': '#1D1160', 'POR': '#E03A3E', 'SAC': '#6F2DA8', 'SAS': '#C4CED4',
        'TOR': '#CE1141', 'UTA': '#002B5C', 'WAS': '#002B5C'}

models_dict = pickle.load(open('model_results/models_dict.pickle', 'rb'))

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

xgb = pickle.load(open('model_results/xgboost_model.pickle', 'rb'))
# print xgb features
print(xgb.feature_importances_)
print(xgb.get_booster().feature_names)

def add_win_prob(df):
    for idx, row in df.iterrows():
        x = row[X_FEATURES]
        prob = predict(x, models_dict)
        df.loc[idx, 'home_win_prob'] = round(prob, 3)
    return df

def add_win_prob_xgb(df):
    df['home_win_prob'] = xgb.predict_proba(df[X_FEATURES])[:, 1]
    return df

def make_plot(df):
    '''
    the x axis is time elapsed: it should go from 0 to max(48, max(df['time_elapsed']))
    the y axis is home win probability, it should go from 0 to 1

    '''
    df['away_score'] = df['away_score'].astype(int)
    df['home_score'] = df['home_score'].astype(int)
    df['score_string'] = df['home_score'].astype(str) + ' - ' + df['away_score'].astype(str)
    df['win_prob_string'] = (100 * df['home_win_prob']).round(1).astype(str) + '%'
    # could also add event string
    winning_team = df.iloc[-1]['home_team_name'] if df.iloc[-1]['home_win_prob'] >= .5 else df.iloc[-1]['away_team_name']


    home_name = df['home_team_name'].iloc[0]
    away_name = df['away_team_name'].iloc[0]
    home_score = df['home_score'].iloc[-1]
    away_score = df['away_score'].iloc[-1]
    current_prob = df['home_win_prob'].iloc[-1]
    current_time = df['string_time_in_period'].iloc[-1]
    current_period = df['period'].iloc[-1]
    if current_period < 5:
        period_string = 'Q' + str(current_period)
    elif current_period >= 5:
        period_string = 'OT' + str(current_period - 4)
    time_string = period_string + ' ' + current_time
    score_string = str(home_score) + ' - ' + str(away_score)
    vs_string = home_name +  ' ' + 'vs.' + ' ' + away_name 
    current_prob_string = 'Current ' + home_name + ' Win Probability: ' + str(round(100 * current_prob, 1)) + '%'
    title = vs_string + '<br>' + score_string
    ylabel = home_name + ' Win Probability'

    data = []
    x = [0, 12, 24, 36, 48, 53, 58, 63, 68]
    xticks = ['1Q', '2Q', '3Q', '4Q', 'Final', 'Final OT1', 'Final OT2', 'Final OT3', 'Final OT4']
    if max(df['time_elapsed']) <= 48:
        x = x[:5]
        xticks = xticks[:5]
    elif max(df['time_elapsed']) <= 53:
        x = x[:6]
        xticks = xticks[:6]
    elif max(df['time_elapsed']) <= 58:
        x = x[:7]
        xticks = xticks[:7]
    elif max(df['time_elapsed']) <= 63:
        x = x[:8]
        xticks = xticks[:8]
    elif max(df['time_elapsed']) <= 68:
        x = x[:9]
        xticks = xticks[:9]
    
    for i in range(len(x)):
        data.append([x[i], xticks[i], np.nan])
    for i, row in df.iterrows():
        data.append([row['time_elapsed'], row['string_time_in_period'], row['home_win_prob'], row['score_string'], row['win_prob_string'], row['event']])

    df = pd.DataFrame(data, columns=['time_elapsed', 'time_string', 'home_win_prob', 'score_string', 'win_prob_string', 'event'])
    df = df.sort_values(by=['time_elapsed'])
    df = df.reset_index(drop=True)
    df = df.interpolate(method='linear', limit_direction='backward')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time_elapsed'], y=df['home_win_prob'], mode='lines', name='lines', line=dict(color=colors[winning_team]),
                            customdata=df[['time_string', 'score_string', 'event', 'win_prob_string']], hovertemplate='%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]}<br>%{customdata[3]}<extra></extra>'))

    fig.update_layout(
        template='plotly_dark',
        title=title,
        xaxis_title="Time",
        yaxis_title=ylabel,
        xaxis=dict(
            tickmode='array',
            tickvals=x,
            ticktext=xticks,
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=['0%', '25%', '50%', '75%', '100%'],
        ),
        # y lim from 0 to 1
        yaxis_range=[0, 1],
        # add some margin to bottom and left
        margin=dict(l=50, b=50),
        xaxis_range = [-1, max(48, max(df['time_elapsed'])) + 1],
        
        # hovermode='x unified',
        # remove legend
        showlegend=False,

        # hover should show time and win probability
        # remove background color
        # plot_bgcolor='rgba(0,0,0,0)',



    )
    
    return fig


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
    num_overtimes = row['period'] - 5
    if row['period'] < 5:
        return (row['period'] - 1) * 12 + 12 - row['time_in_period']
    else:
        return 48 + num_overtimes * 5 + 5 - row['time_in_period']

def time_in_period(row):
    minutes = row['Time'].split(':')[0]
    seconds = row['Time'].split(':')[1]
    return int(minutes) + float(seconds)/60

def find_event(row):
    if not isinstance(row['HomeEvent'], float):
        return str(row['HomeEvent'])
    elif not isinstance(row['AwayEvent'], float):
        return str(row['AwayEvent'])
    else:
        return ''


def format_pbp_df_for_model(df):
    df[['HomeScore', 'AwayScore']] = df[['HomeScore', 'AwayScore']].fillna(method='ffill')
    df[['HomeScore', 'AwayScore']] = df[['HomeScore', 'AwayScore']].fillna(0)
    df[['HomeScore', 'AwayScore']] = df[['HomeScore', 'AwayScore']].astype(int)
    df['home_team_name'] = df['HomeName']
    df['away_team_name'] = df['AwayName']
    df['home_score'] = df['HomeScore'].astype(int)
    df['away_score'] = df['AwayScore'].astype(int)
    df['home_margin'] = df['home_score'] - df['away_score']
    df['period'] = df['Period'].astype(int)
    df['time_in_period'] = df.apply(lambda row: time_in_period(row), axis=1)
    df['time_in_period_minutes'] = (60 * df['time_in_period']) // 60
    df['time_in_period_minutes'] = df['time_in_period_minutes'].astype(int)
    df['time_in_period_seconds'] = (60 * df['time_in_period']) % 60
    df['time_in_period_seconds'] = df['time_in_period_seconds'].astype(int)
    df['string_time_in_period'] = df.apply(time_string, axis=1)
    df['home_margin_diff'] = df['home_margin'].diff()
    df['home_margin_diff'] = df['home_margin_diff'].fillna(0)
    df['home_margin_diff_2'] = df['home_margin_diff'].diff()
    df['home_margin_diff_2'] = df['home_margin_diff_2'].fillna(0)
    df['time_elapsed'] = round(df.apply(time_elapsed, axis=1), 2)
    df['time_remaining'] = df.apply(time_remaining, axis=1)
    df['event'] = df.apply(lambda row: find_event(row), axis=1)
    df = add_possessions(df)
    df = add_possession_by_team(df)
    df['home_possession'] = df['HomePossession']
    df.rename(columns={'AwayName': 'away_name', 'HomeName': 'home_name'}, inplace=True)
    df = experimental_format(df)
    return df

def experimental_format(pbp_df):
    pbp_df['fts_remaining'] = pbp_df.apply(add_fts, axis=1)
    pbp_df['foul'] = pbp_df.apply(lambda row: 'foul' in row['event'].lower(), axis=1)
    pbp_df['turnover'] = pbp_df.apply(lambda row: 'turnover' in row['event'].lower(), axis=1)
    pbp_df['steal'] = pbp_df.apply(lambda row: 'steal' in row['event'].lower() or 'steal' in row['HomeEvent'].lower(), axis=1)
    pbp_df['block'] = pbp_df.apply(lambda row: 'block' in row['event'].lower() or 'block' in row['HomeEvent'].lower(), axis=1)
    pbp_df['timeout'] = pbp_df.apply(lambda row: 'timeout' in row['event'].lower(), axis=1)
    pbp_df['offensive_foul'] = pbp_df.apply(lambda row: 'offensive foul' in row['event'].lower(), axis=1)
    pbp_df['defensive_foul'] = pbp_df.apply(lambda row: 'defensive foul' in row['event'].lower(), axis=1)
    pbp_df['offensive_rebound'] = pbp_df.apply(lambda row: 'offensive rebound' in row['event'].lower(), axis=1)
    pbp_df['defensive_rebound'] = pbp_df.apply(lambda row: 'defensive rebound' in row['event'].lower(), axis=1)
    return pbp_df
    

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

def get_excitement_index(df):
    '''
    computes the average absolute value of the derivative of win probability with respect to time elapsed
    '''
    # insert time intervals at every .01 minutes
    for time in np.arange(0, max(df['time_elapsed']) + .01, .01):
        if time not in df['time_elapsed'].values:
            df = df.append({'time_elapsed': time}, ignore_index=True)
    df = df.sort_values(by='time_elapsed')
    df = df.interpolate(method='linear', limit_direction='backward')
    df['win_prob_diff'] = df['home_win_prob'].diff().abs()

    return df['win_prob_diff'].sum() / max(df['time_elapsed']) * 100

def get_dominance_index(df):
    '''
    dominance for the home team
    this function returns the integral of the home win probability with respect to time elapsed
    '''
    # insert time intervals at every .01 minutes
    for time in np.arange(0, max(df['time_elapsed']) + .01, .01):
        if time not in df['time_elapsed'].values:
            df = df.append({'time_elapsed': time}, ignore_index=True)
    df = df.sort_values(by='time_elapsed')
    df = df.interpolate(method='linear', limit_direction='backward')
    df['time_elapsed_diff'] = df['time_elapsed'].diff()
    df['home_win_prob_adv'] = df['home_win_prob']
    df['dominance'] = df['home_win_prob_adv'] * df['time_elapsed_diff']
    # technically a right riemann sum, would like to do a trapezoidal sum
    return df['dominance'].sum() / max(df['time_elapsed']) * 100


def get_tension_index(df):
    '''
    returns the average entropy of the win probability
    '''
    # insert time intervals at every .01 minutes
    for time in np.arange(0, max(df['time_elapsed']) + .01, .01):
        if time not in df['time_elapsed'].values:
            df = df.append({'time_elapsed': time}, ignore_index=True)
    df = df.sort_values(by='time_elapsed')  
    df = df.interpolate(method='linear', limit_direction='backward')
    df['time_elapsed_diff'] = df['time_elapsed'].diff()
    df['win_prob_entropy'] = - df['home_win_prob'] * np.log(df['home_win_prob']) - (1 - df['home_win_prob']) * np.log(1 - df['home_win_prob'])
    # weight it by the time elapsed
    df['win_prob_entropy'] = df['win_prob_entropy'] * df['time_elapsed_diff']
    # drop where entropy is na
    df = df.dropna(subset=['win_prob_entropy'])
    return df['win_prob_entropy'].sum() / max(df['time_elapsed']) * 100

def main():
    data_dict = {}
    game_indices_df = pd.DataFrame(columns=['game', 'team', 'opponent', 'excitement_index', 'dominance_index', 'tension_index'])
    dir = 'pbp_data/with_odds/2023/'
    for filename in os.listdir(dir):
        if filename.endswith('.csv'):
            df = pd.read_csv(dir + filename)
            print(df.head())
            # check if empty df
            if df.empty:
                print('empty df:', filename)
                continue
            game_id = filename[:-4]

            df = format_pbp_df_for_model(df)
            data_dict[game_id] = df
            df = add_win_prob_xgb(df)
            df.iloc[-1, df.columns.get_loc('home_win_prob')] = int(df.iloc[-1, df.columns.get_loc('home_score')] > df.iloc[-1, df.columns.get_loc('away_score')])
            excitement_index = get_excitement_index(df)
            dominance_index = get_dominance_index(df)
            tension_index = get_tension_index(df)
            # add to dataframe
            game_indices_df = game_indices_df.append({'game': game_id, 'team': df.iloc[0, df.columns.get_loc('home_name')], 'opponent': df.iloc[0, df.columns.get_loc('away_name')], 'excitement_index': excitement_index, 'dominance_index': dominance_index, 'tension_index': tension_index}, ignore_index=True)
            print()
            print(game_id, 'excitement_index:', round(excitement_index, 2), 'dominance_index:', round(dominance_index, 2), 'tension_index:', round(tension_index, 2))
            print()
            print(game_indices_df.sort_values(by='excitement_index', ascending=False).head(10))
            print()
            print(game_indices_df.sort_values(by='dominance_index', ascending=False).head(10))
            print()
            print(game_indices_df.sort_values(by='tension_index', ascending=False).head(10))
            print()


            fig = make_plot(df)
            # fig.show()
            fig.write_html('plots/' + str(game_id) + '.html')
    game_indices_df.to_csv('game_indices.csv', index=False)
    #         # make first insance of each team score 0
    #         df = format_pbp_df_for_model(df)
    #         df = add_win_prob(df)
    #         df.iloc[-1, df.columns.get_loc('home_win_prob')] = int(df.iloc[-1, df.columns.get_loc('home_score')] > df.iloc[-1, df.columns.get_loc('away_score')])
    #         fig = make_plot(df)
    #         # fig.show()
    #         fig.write_html('plots/' + filename[:-4] + '.html')

    # pbp_dict = pickle.load(open('training_data/pbp_data_for_xgb.pickle', 'rb'))
    # for game_id, df in pbp_dict.items():
    #     df['home_margin'] = df['home_margin'].fillna(method='ffill')
    #     df['home_margin'] = df['home_margin'].fillna(0)
    #     df['home_score'] = df['home_score'].fillna(method='ffill')
    #     df['home_score'] = df['home_score'].fillna(0)
    #     df['away_score'] = df['away_score'].fillna(method='ffill')
    #     df['away_score'] = df['away_score'].fillna(0)
    #     df = experimental_format(df)
    #     df = add_win_prob_xgb(df)
    #     excitement_index = get_excitement_index(df)
    #     dominance_index = get_dominance_index(df)
    #     tension_index = get_tension_index(df)
    #     # add to dataframe
    #     game_indices_df = game_indices_df.append({'game': game_id, 'team': df.iloc[0, df.columns.get_loc('home_name')], 'opponent': df.iloc[0, df.columns.get_loc('away_name')], 'excitement_index': excitement_index, 'dominance_index': dominance_index, 'tension_index': tension_index}, ignore_index=True)
    #     print()
    #     print(game_id, 'excitement_index:', round(excitement_index, 2), 'dominance_index:', round(dominance_index, 2), 'tension_index:', round(tension_index, 2))
    #     print()
    #     print(game_indices_df.sort_values(by='excitement_index', ascending=False).head(10))
    #     print()
    #     print(game_indices_df.sort_values(by='dominance_index', ascending=False).head(10))
    #     print()
    #     print(game_indices_df.sort_values(by='tension_index', ascending=False).head(10))
    #     print()

    #     df.rename(columns={'away_name': 'away_team_name', 'home_name': 'home_team_name'}, inplace=True)
    #     df['time_in_period_minutes'] = (60 * df['time_in_period']) // 60
    #     df['time_in_period_minutes'] = df['time_in_period_minutes'].astype(int)
    #     df['time_in_period_seconds'] = (60 * df['time_in_period']) % 60
    #     df['time_in_period_seconds'] = df['time_in_period_seconds'].astype(int)
    #     df['string_time_in_period'] = df.apply(time_string, axis=1)
    #     df['event'] = df.apply(find_event, axis=1)
    #     df.iloc[-1, df.columns.get_loc('home_win_prob')] = int(df.iloc[-1, df.columns.get_loc('home_score')] > df.iloc[-1, df.columns.get_loc('away_score')])
    #     fig = make_plot(df)
    #     # fig.show()
    #     fig.write_html('plots/' + str(game_id) + '.html')
    # game_indices_df.to_csv('game_indices.csv', index=False)

if __name__ == '__main__':
    # xgb = pickle.load(open('model_results/xgboost_model.pickle', 'rb'))
    # # get feature importance
    # print(xgb.feature_importances_)
    # main()
    pass