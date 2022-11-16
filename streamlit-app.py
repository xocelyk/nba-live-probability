import streamlit as st
from scrape_nba_dot_com.utils import *
from datetime import datetime
import pickle
import pandas as pd
from plotly import graph_objects as go


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

def find_today_games():
    today_pbp_dict = {}
    today_odds_dict = scrape_today_odds()
    schedule = scrape_schedule()
    today_date = datetime.today().date()
    schedule['date'] = pd.to_datetime(schedule['game_date']).dt.date
    today_schedule = schedule[schedule['date'] == today_date]

    for game_id, game in today_schedule.iterrows():
        if game_id in today_odds_dict:
            game_odds = today_odds_dict[game_id]
            game['home_ml'] = game_odds['home_ml']
            game['away_ml'] = game_odds['away_ml']
            game['home_spread'] = game_odds['home_spread']
            game['home_spread_odds'] = game_odds['home_spread_odds']
            game['away_spread'] = game_odds['away_spread']
            game['away_spread_odds'] = game_odds['away_spread_odds']
        else:
            today_pbp_dict[game_id] = None
            continue
        pbp_data = parse_game(schedule, game_id)
        if pbp_data is None:
            # game has not happened yet
            today_pbp_dict[game_id] = None
        else:
            pbp_data['home_ml'] = game['home_ml']
            pbp_data['away_ml'] = game['away_ml']
            pbp_data['home_spread'] = -float(game['home_spread'])
            pbp_data['home_spread_odds'] = game['home_spread_odds']
            pbp_data['away_spread'] = -float(game['away_spread'])
            pbp_data['away_spread_odds'] = game['away_spread_odds']
            today_pbp_dict[game_id] = pbp_data
    return today_pbp_dict


def time_string(row):
    minutes = str(row['time_in_period_minutes'])
    seconds = str(row['time_in_period_seconds'])
    if len(seconds) == 1:
        seconds = '0' + seconds
    return minutes + ':' + seconds


def format_pbp_df_for_model(df):
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
    df['home_margin_diff'] = df['home_margin'].diff()
    df['home_margin_diff'] = df['home_margin_diff'].fillna(0)
    df['time_elapsed'] = df.apply(time_elapsed, axis=1)
    df['time_remaining'] = df.apply(time_remaining, axis=1)
    df = df[['period', 'time_elapsed', 'string_time_in_period', 'time_remaining', 'home_margin', 'home_margin_diff', 'home_close_spread', 'home_team_name', 'away_team_name', 'home_score', 'away_score']]
    return df

def make_plot(df):
    '''
    the x axis is time elapsed: it should go from 0 to max(48, max(df['time_elapsed']))
    the y axis is home win probability, it should go from 0 to 1

    '''

    home_name = df['home_team_name'].iloc[0]
    away_name = df['away_team_name'].iloc[0]
    home_score = df['home_score'].iloc[-1]
    away_score = df['away_score'].iloc[-1]
    current_prob = df['home_win_prob'].iloc[-1]

    title = home_name +  ' ' + str(home_score) + ' - ' + str(away_score) + ' ' + away_name + '\t' + '('  + str(round(100 * current_prob, 1)) + '%' + ' - ' + str(round(100 * (1 - current_prob), 1)) + '%' + ')'
    ylabel = home_name + ' Win Probability'

    data = []
    x = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 53, 58, 63, 68]
    xticks = ['1Q 12:00', '1Q 8:00', '1Q 4:00', '2Q 12:00', '2Q 8:00', '2Q 4:00', '3Q 12:00', '3Q 8:00', '3Q 4:00', '4Q 12:00', '4Q 8:00', '4Q 4:00', '4Q 0:00', 'OT1 5:00', 'OT2 5:00', 'OT3 5:00', 'OT4 5:00']
    if max(df['time_elapsed']) < 48:
        x = x[:13]
        xticks = xticks[:13]
    elif max(df['time_elapsed']) < 53:
        x = x[:14]
        xticks = xticks[:14]
    elif max(df['time_elapsed']) < 58:
        x = x[:15]
        xticks = xticks[:15]
    elif max(df['time_elapsed']) < 63:
        x = x[:16]
        xticks = xticks[:16]
    elif max(df['time_elapsed']) < 68:
        x = x[:17]
        xticks = xticks[:17]
    
    for i in range(len(x)):
        data.append([x[i], xticks[i], np.nan])
    for i, row in df.iterrows():
        data.append([row['time_elapsed'], row['string_time_in_period'], row['home_win_prob']])

    df = pd.DataFrame(data, columns=['time_elapsed', 'time_string', 'home_win_prob'])
    df = df.sort_values(by=['time_elapsed'])
    df = df.reset_index(drop=True)
    df = df.interpolate(method='linear', limit_direction='backward')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time_elapsed'], y=df['home_win_prob'], mode='lines', name='lines'))
    fig.update_layout(
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
        yaxis_range=[0, 1]
    )
    
    return fig

def figlist():
    today_pbp_dict = find_today_games()
    format_pbp_dict = {}
    today_pbp_dict = {k: v for k, v in today_pbp_dict.items() if v is not None}
    for game_id, game_data in today_pbp_dict.items():
        format_pbp_dict[game_id] = format_pbp_df_for_model(game_data)
    
    fig_list = []
    # load model dict from pickle
    filename = 'models_dict.pickle'
    model_dict = pickle.load(open(filename, 'rb'))
    from live_probability_model import predict
    for game_id, game_df in format_pbp_dict.items():
        for idx, row in game_df.iterrows():
            x = row[['time_remaining', 'home_margin', 'home_margin_diff', 'home_close_spread']]
            prob = predict(x, model_dict)
            game_df.loc[idx, 'home_win_prob'] = prob

        plot = make_plot(game_df)
        fig_list.append(plot)
    
    return fig_list

def main():
    st.title('NBA Live Win Probability')
    st.write('here')
    fig_list = figlist()
    for fig in fig_list:
        st.plotly_chart(fig, use_container_width=True)
    st.write('finished')
        
if __name__ == '__main__':
    main()
 

