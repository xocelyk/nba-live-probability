import streamlit as st
from scrape_nba_dot_com.utils import *
from datetime import datetime
import pickle
import pandas as pd
from plotly import graph_objects as go
from live_probability_model import predict
import time


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

# @st.cache
def get_schedule(today_date):
    schedule = scrape_schedule()
    schedule['date'] = pd.to_datetime(schedule['game_date']).dt.date
    today_schedule = schedule[schedule['date'] == today_date]
    yesterday_schedule = schedule[schedule['date'] == today_date - pd.Timedelta(days=1)]
    today_schedule = today_schedule.append(yesterday_schedule)
    return today_schedule

def find_today_games(today_schedule, today_odds_dict):
    today_pbp_dict = {}
    for game_id, game in today_schedule.iterrows():
        if game_id in today_odds_dict:
            game_odds = today_odds_dict[game_id]
            try:
                game['home_ml'] = game_odds['home_ml']
                game['away_ml'] = game_odds['away_ml']
            except:
                game['home_ml'] = None
                game['away_ml'] = None
            try:
                game['home_spread'] = game_odds['home_spread']
                game['home_spread_odds'] = game_odds['home_spread_odds']
                game['away_spread'] = game_odds['away_spread']
                game['away_spread_odds'] = game_odds['away_spread_odds']
            except:
                game['home_spread'] = None
                game['home_spread_odds'] = None
                game['away_spread'] = None
                game['away_spread_odds'] = None
        else:
            today_pbp_dict[game_id] = None
            continue
        pbp_data = parse_game(today_schedule, game_id)
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
    df['home_margin_diff_2'] = df['home_margin_diff'].diff()
    df['home_margin_diff_2'] = df['home_margin_diff_2'].fillna(0)
    df['time_elapsed'] = df.apply(time_elapsed, axis=1)
    df['time_remaining'] = df.apply(time_remaining, axis=1)
    df['event'] = df['description']
    df = df[['period', 'time_elapsed', 'string_time_in_period', 'time_remaining', 'home_margin', 'home_margin_diff', 'home_margin_diff_2', 'home_close_spread', 'home_team_name', 'away_team_name', 'home_score', 'away_score', 'event']]
    return df

def make_plot(df):
    '''
    the x axis is time elapsed: it should go from 0 to max(48, max(df['time_elapsed']))
    the y axis is home win probability, it should go from 0 to 1

    '''

    # colors = {'ATL': '#E03A3E', 'BRK': '#5F6264', 'BOS': '#007A33', 'CHO': '#00FFFF', 'CHI': '#CE1141', 'CLE': '#860038', 'DAL': '#00538C', 'DEN': '#0E2240', 'DET': '#C8102E',
    #     'GSW': '#006BB6', 'HOU': '#CE1141', 'IND': '#002D62', 'LAC': '#C8102E', 'LAL': '#552583', 'MEM': '#5D76A9', 'MIA': '#98002E', 'MIL': '#00471B', 'MIN': '#32CD32',
    #     'NOP': '#0C2340', 'NYK': '#FFA500', 'OKC': '#007AC1', 'ORL': '#0077C0', 'PHI': '#006BB6', 'PHO': '#1D1160', 'POR': '#E03A3E', 'SAC': '#5A2D81', 'SAS': '#C4CED4',
    #     'TOR': '#CE1141', 'UTA': '#002B5C', 'WAS': '#002B5C'}

    colors = {'Atlanta Hawks': '#0077C0', 'Brooklyn Nets': '#5F6264', 'Boston Celtics': '#007A33', 'Charlotte Hornets': '#00FFFF', 'Chicago Bulls': '#CE1141', 'Cleveland Cavaliers': '#860038', 'Dallas Mavericks': '#00538C', 'Denver Nuggets': '#0E2240', 'Detroit Pistons': '#C8102E', \
    'Golden State Warriors': '#006BB6', 'Houston Rockets': '#CE1141', 'Indiana Pacers': '#002D62', 'Los Angeles Clippers': '#C8102E', 'Los Angeles Lakers': '#552583', 'Memphis Grizzlies': '#5D76A9', 'Miami Heat': '#98002E', 'Milwaukee Bucks': '#00471B', 'Minnesota Timberwolves': '#32CD32', \
    'New Orleans Pelicans': '#0062CC', 'New York Knicks': '#FFA500', 'Oklahoma City Thunder': '#007AC1', 'Orlando Magic': '#0077C0', 'Philadelphia 76ers': '#006BB6', 'Phoenix Suns': '#1D1160', 'Portland Trail Blazers': '#E03A3E', 'Sacramento Kings': '#5A2D81', 'San Antonio Spurs': '#C4CED4', \
    'Toronto Raptors': '#CE1141', 'Utah Jazz': '#002B5C', 'Washington Wizards': '#0062CC'}



    df['score_string'] = df['home_score'].astype(str) + '-' + df['away_score'].astype(str)
    df['win_prob_string'] = (100 * df['home_win_prob']).round(1).astype(str) + '%'
    home_name = df['home_team_name'].iloc[0]
    away_name = df['away_team_name'].iloc[0]
    winning_team = df.iloc[-1]['home_team_name'] if df.iloc[-1]['home_win_prob'] >= .5 else df.iloc[-1]['away_team_name']
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
    title = vs_string + ' (' + time_string + ')' + '<br>' + score_string + '<br>' + '<sup>' + current_prob_string + '</sup>'
    ylabel = home_name + ' Win Probability'

    df['home_win_prob'] = round(df['home_win_prob'], 3)
    df['time_elapsed'] = round(df['time_elapsed'], 2)

    data = []
    x = [0, 12, 24, 36, 48, 53, 58, 63, 68]
    xticks = ['1Q 12:00', '2Q 12:00', '3Q 12:00', '4Q 12:00', '4Q 0:00', 'OT1 0:00', 'OT2 0:00', 'OT3 0:00', 'OT4 0:00']
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
        data.append([row['time_elapsed'], row['string_time_in_period'], row['home_win_prob'], row['score_string'], row['event'], row['win_prob_string']])

    df = pd.DataFrame(data, columns=['time_elapsed', 'string_time_in_period', 'home_win_prob', 'score_string', 'event', 'win_prob_string'])
    df = df.sort_values(by=['time_elapsed'])
    df = df.reset_index(drop=True)
    df = df.interpolate(method='linear', limit_direction='backward')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time_elapsed'], y=df['home_win_prob'], mode='lines', name='lines', line=dict(color=colors[winning_team]),
                            customdata=df[['score_string', 'event', 'win_prob_string']], hovertemplate='%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>'))

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
        
        hovermode='closest',
        # remove legend
        showlegend=False,

        # make grid lines light grey
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        xaxis_gridcolor='#404040',
        yaxis_gridcolor='#404040',




    )
    
    return fig

# @st.cache
def figlist(dfs_list):
    figlist = []
    for game_df in dfs_list:
        plot = make_plot(game_df)
        figlist.append(plot)
    return figlist

def predict_game(format_pbp_dict):
    game_dfs_list = []
    filename = 'model_results/models_dict.pickle'
    model_dict = pickle.load(open(filename, 'rb'))
    for game_id, game_df in format_pbp_dict.items():
        for idx, row in game_df.iterrows():
            x = row[['time_remaining', 'home_margin', 'home_margin_diff', 'home_close_spread']]
            prob = predict(x, model_dict)
            game_df.loc[idx, 'home_win_prob'] = prob
        game_dfs_list.append(game_df)
    return game_dfs_list

def main():
    st.set_page_config(layout="wide")
    st.title('NBA Live Win Probability')

    placeholder = st.empty()

    while True:
        today_odds_dict = scrape_today_odds()
        today_schedule = get_schedule(datetime.today().date())
        today_pbp_dict = find_today_games(today_schedule, today_odds_dict)
        format_pbp_dict = {}
        today_pbp_dict = {k: v for k, v in today_pbp_dict.items() if v is not None}
        for game_id, game_data in today_pbp_dict.items():
            format_pbp_dict[game_id] = format_pbp_df_for_model(game_data)
        dfs_list = predict_game(format_pbp_dict)
        fig_list = figlist(dfs_list)

        with placeholder.container():
            col1, col2 = st.columns(2)
            for i, fig in enumerate(fig_list):
                if i % 2 == 0:
                    col1.plotly_chart(fig)
                else:
                    col2.plotly_chart(fig)
        time.sleep(10)

        
if __name__ == '__main__':
    main()
 

