import streamlit as st
from scrape_nba_dot_com.utils import *
from datetime import datetime
import pickle
import pandas as pd
from plotly import graph_objects as go
from live_probability_model import predict
import time
from format_pbp2.format_nba_dot_com_pbp import format_pbp_df
from probability_archive_report import get_dominance_rankings
from plot import get_tension_index, get_excitement_index, get_dominance_index
import dill


X_FEATURES = ['time_remaining', 'home_margin', 'home_possession', 'home_close_spread', 'fts_remaining', 'foul', 'turnover', 'steal', 'block', 'timeout', 'offensive_foul', 'defensive_foul', 'offensive_rebound', 'defensive_rebound']

colors = {'Atlanta Hawks': '#E03A3E', 'Brooklyn Nets': '#5F6264', 'Boston Celtics': '#007A33', 'Charlotte Hornets': '#00FFFF', 'Chicago Bulls': '#CE1141', 'Cleveland Cavaliers': '#860038', 'Dallas Mavericks': '#00538C', 'Denver Nuggets': '#FDB827', 'Detroit Pistons': '#C8102E',
        'Golden State Warriors': '#006BB6', 'Houston Rockets': '#CE1141', 'Indiana Pacers': '#002D62', 'LA Clippers': '#C8102E', 'Los Angeles Lakers': '#552583', 'Memphis Grizzlies': '#5D76A9', 'Miami Heat': '#98002E', 'Milwaukee Bucks': '#00471B', 'Minnesota Timberwolves': '#32CD32',
        'New Orleans Pelicans': '#0C2340', 'New York Knicks': '#FFA500', 'Oklahoma City Thunder': '#007AC1', 'Orlando Magic': '#0077C0', 'Philadelphia 76ers': '#006BB6', 'Phoenix Suns': '#1D1160', 'Portland Trail Blazers': '#E03A3E', 'Sacramento Kings': '#6F2DA8', 'San Antonio Spurs': '#C4CED4',
        'Toronto Raptors': '#CE1141', 'Utah Jazz': '#002B5C', 'Washington Wizards': '#002B5C'}

names_to_abbrs = {'Atlanta Hawks': 'ATL', 'Brooklyn Nets': 'BRK', 'Boston Celtics': 'BOS', 'Charlotte Hornets': 'CHO', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
                'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
                'LA Clippers': 'LAC', 'LA Lakers': 'LAL', 'Los Angeles Lakers': 'LAL', 'Los Angeles Clippers': 'LAC', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
                'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHO',
                'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'}

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

def get_schedule(today_date):
    schedule = scrape_schedule()
    # change date to datetime object but only date
    schedule['date'] = pd.to_datetime(schedule['game_date']).dt.date
    today_schedule = schedule[schedule['date'] == today_date]
    # handling timezones
    yesterday_schedule = schedule[schedule['date'] == today_date - pd.Timedelta(days=1)]
    today_schedule = pd.concat([today_schedule, yesterday_schedule])
    return today_schedule

def find_today_games(today_schedule, today_odds_dict):
    today_pbp_dict = {}
    for game_id, game in today_schedule.iterrows():
        if today_odds_dict is None:
            game['home_ml'] = 100
            game['away_ml'] = 100
            game['home_spread'] = 0
            game['home_spread_odds'] = 100
            game['away_spread'] = 0
            game['away_spread_odds'] = 100
        elif game_id in today_odds_dict:
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
    today_pbp_dict = {k: v for k, v in today_pbp_dict.items() if v is not None}
    return today_pbp_dict

def time_string(row):
    minutes = str(row['time_in_period_minutes'])
    seconds = str(row['time_in_period_seconds'])
    if len(seconds) == 1:
        seconds = '0' + seconds
    return minutes + ':' + seconds

def format_pbp_df_for_model(df):
    # TODO: double check this works before deploying
    df = format_pbp_df(df)
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
    title = vs_string + ' (' + time_string + ')' + '<br>' + score_string + '<br>' + '<sup>' + current_prob_string + '</sup>'
    ylabel = home_name + ' Win Probability'

    data = []
    x = [0, 12, 24, 36, 48, 53, 58, 63, 68]
    xticks = ['1Q', '2Q', '3Q', '4Q', 'End Regulation', 'End OT1', 'End OT2', 'End OT3', 'End OT4']
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
        yaxis_range=[0, 1],
        margin=dict(l=50, b=50),
        xaxis_range = [-1, max(48, max(df['time_elapsed'])) + 1],
        
        showlegend=False,
    )
    
    return fig

def figlist(dfs_list):
    figlist = []
    for game_df in dfs_list:
        plot = make_plot(game_df)
        # make plot a little bigger
        plot.update_layout(height=600, width=1000)
        figlist.append(plot)
    return figlist

# TODO: predict with xgb model and format data differently for feature extraction
def predict_game(game_df):
    game_dfs_dict = {}
    filename = 'model_results/xgboost_model.pickle'
    model = dill.load(open(filename, 'rb'))
    game_df['home_win_prob'] = model.predict_proba(game_df[X_FEATURES])[:, 1]
    if game_df['description'].iloc[-1] == 'Game End':
        game_df.iloc[-1, game_df.columns.get_loc('home_win_prob')] = int(game_df['scoreHome'].iloc[-1] > game_df['scoreAway'].iloc[-1])
    return game_df

@ st.cache(allow_output_mutation=True)
def get_archive_table():
    '''
    going to have to update the archive manually on the other script
    '''
    plot_dict = {}
    filename = '2023_archive.pickle'
    archive_data = []
    archive_dict = dill.load(open(filename, 'rb'))
    for boxscore_id, game_data in archive_dict.items():
        df = game_data['df']
        plot = game_data['plot']
        year = boxscore_id[:4]
        month = boxscore_id[4:6]
        day = boxscore_id[6:8]
        date = '{}-{}-{}'.format(year, month, day)
        excitement_index = game_data['excitement_index']
        tension_index = game_data['tension_index']
        dominance_index = game_data['dominance_index']
        home_team = df.iloc[0]['home_team_name']
        away_team = df.iloc[0]['away_team_name']
        home_score = df.iloc[-1]['home_score']
        away_score = df.iloc[-1]['away_score']
        archive_data.append([boxscore_id, date, home_team, away_team, home_score, away_score, excitement_index, tension_index, dominance_index])
        plot_dict[boxscore_id] = plot
    archive_df = pd.DataFrame(archive_data, columns=['boxscore_id', 'Date', 'Home', 'Away', 'Home Score', 'Away Score', 'Excitement', 'Tension', 'Dominance'])
    archive_df = archive_df.sort_values(by=['Date', 'Home'], ascending=True)
    archive_df.index = archive_df.apply(lambda row: row['Date'].split('-')[0] + '-' + row['Date'].split('-')[1] + '-' + row['Date'].split('-')[2] + ' ' +  row['Home'] + ' vs. ' + row['Away'], axis=1)
    # reindex plot dict by archive_df index
    plot_dict = {k: plot_dict[v] for k, v in zip(archive_df.index, archive_df['boxscore_id'])}
    return archive_df, plot_dict

def live_probability_page():
    # TODO: cache the formatting process
    import time
    # st.write("I've lost my free pre-game odds provider, which is fundamental to the win probability model. Until I find a new data source, the live page is down, but I encourage you to check out the archive in the side menu!")
    page_start_time = time.time()
    st.title('NBA Live Win Probability')

    sleep_time = 10
    placeholder = st.empty()

    cache_dfs_dict = {}
    first_run = True

    cache_today_odds_dict = {}
    cache_today_schedule = {}
    counter = 0
    while True:
        process_1_time = time.time()
        if counter % 10 == 0:
            try:
                today_odds_dict = scrape_today_odds()
            except:
                today_odds_dict = None
            today_schedule = get_schedule(datetime.today().date())
            cache_today_odds_dict = today_odds_dict
            cache_today_schedule = today_schedule
        else:
            today_odds_dict = cache_today_odds_dict
            today_schedule = cache_today_schedule
        today_pbp_dict = find_today_games(today_schedule, today_odds_dict)
        if len(today_pbp_dict) == 0:
            st.write('No games right now. Check back later.')
            break
        format_pbp_dict = {}

        matchup_list = []
        home_abbr_list = []

        for game_id, game_data in today_pbp_dict.items():
            format_pbp_dict[game_id] = format_pbp_df_for_model(game_data)
            matchup_list.append((game_data['home_team_name'], game_data['away_team_name']))
            home_abbr_list.append(names_to_abbrs[game_data['home_team_name'].values[0]])


        dfs_dict = {}
        for game_id, game_data in today_pbp_dict.items():
            if game_id in cache_dfs_dict:
                new_indices = game_data.index.difference(cache_dfs_dict[game_id].index)
                if len(new_indices) > 0:
                    new_format_df = format_pbp_df_for_model(game_data.loc[new_indices])
                    new_pred_df = predict_game(new_format_df)
                    dfs_dict[game_id] = pd.concat([cache_dfs_dict[game_id], new_pred_df])
                    cache_dfs_dict[game_id] = dfs_dict[game_id]
                    matchup_list.append((new_format_df['home_team_name'], new_format_df['away_team_name']))
                    home_abbr_list.append(names_to_abbrs[new_format_df['home_team_name'].values[0]])
                else:
                    dfs_dict[game_id] = cache_dfs_dict[game_id]
                    matchup_list.append((cache_dfs_dict[game_id]['home_team_name'], cache_dfs_dict[game_id]['away_team_name']))
                    home_abbr_list.append(names_to_abbrs[cache_dfs_dict[game_id]['home_team_name'].values[0]])
            else:
                game_df = format_pbp_df_for_model(game_data)
                matchup_list.append((game_df['home_team_name'], game_df['away_team_name']))
                home_abbr_list.append(names_to_abbrs[game_df['home_team_name'].values[0]])
                dfs_dict[game_id] = predict_game(game_df)
                cache_dfs_dict[game_id] = dfs_dict[game_id]


        # TODO: cache the metrics
        cur_win_prob_list = []
        tension_list = []
        excitement_list = []
        dominance_list = []
        for game_id, game_df in dfs_dict.items():
            cur_win_prob_list.append(game_df['home_win_prob'].values[-1])
            tension_list.append(get_tension_index(game_df))
            excitement_list.append(get_excitement_index(game_df))
            dominance_list.append(get_dominance_index(game_df))
            
        fig_list = figlist(dfs_dict.values())

        with placeholder.container():
            for i, fig in enumerate(fig_list):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric('Tension', round(tension_list[i], 1))
                col2.metric('Excitement', round(excitement_list[i], 1))
                col3.metric('{} Dominance'.format(home_abbr_list[i]), round(dominance_list[i], 1))
                col4.metric('{} Win Probability'.format(home_abbr_list[i]), round(100 * cur_win_prob_list[i], 1))

                st.plotly_chart(fig)

        time.sleep(sleep_time)
        counter += 1

def archive_page():
    # TODO: write some explanation stuff
    st.title('NBA Live Win Probability Archive')
    archive_df, plot_dict = get_archive_table()
    df_for_show = archive_df.copy()
    df_for_show['Date'] = pd.to_datetime(df_for_show['Date']).dt.date
    df_for_show = df_for_show.sort_values(by=['Date'], ascending=True)
    df_for_show['Final Score'] = df_for_show['Home Score'].astype(str) + '-' + df_for_show['Away Score'].astype(str)
    # round the descriptive stats to 2 decimal places
    df_for_show['Excitement'] = df_for_show['Excitement'].round(4)
    df_for_show['Tension'] = df_for_show['Tension'].round(4)
    df_for_show['Dominance'] = df_for_show['Dominance'].round(4)

    df_for_show = df_for_show[['Date', 'Home', 'Away', 'Final Score', 'Excitement', 'Tension', 'Dominance']]

    st.dataframe(df_for_show.set_index('Date'))

    dict_id = st.selectbox('Select a game', sorted(plot_dict.keys()))
    game = df_for_show.loc[dict_id]

    col1, col2, col3 = st.columns(3)
    # put metrics in each columns
    col1.metric('Excitement', round(game['Excitement'], 1))
    col2.metric('Tension', round(game['Tension'], 1))
    col3.metric('Dominance', round(game['Dominance'], 1))

    plot = plot_dict[dict_id]
    # make plot bigger
    plot.update_layout(height=600, width=1000)
    st.plotly_chart(plot)

def dominance_rankings_page():
    st.title('NBA Dominance Rankings')
    # TODO: write some explanation stuff
    rankings_df = get_dominance_rankings()
    st.dataframe(rankings_df)
        
if __name__ == '__main__':
    # make two different pages
    st.set_page_config(layout="wide")
    page = st.sidebar.selectbox('Select a page', ['Live Win Probability', 'Archive', 'Dominance Rankings'])
    if page == 'Live Win Probability':
        live_probability_page()
    elif page == 'Archive':
        archive_page()
    elif page == 'Dominance Rankings':
        dominance_rankings_page()
 
