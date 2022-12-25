import pandas as pd
import pickle
from format_sportsbookreviews_odds import format_odds_df
from scrape_sr.scrape_pbp import *
from format_pbp2.format_sportsreference_pbp import format as format_pbp
from plot import *
model = pickle.load(open('model_results/xgboost_model.pickle', 'rb'))


# get the odds from sportsbookreviews archive
url = 'https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nba%20odds%202022-23.xlsx'
# read the excel file as a dataframe
year = 2023
df = pd.read_excel(url, sheet_name='Sheet1')
odds_df = format_odds_df(df, year)
# turn into dictionary where index is key and home_close_spread is value
odds_dict = odds_df.set_index('game_id')['home_close_spread'].to_dict()

boxscore_ids = get_boxscore_ids(year)

team_abbr_to_sr_review_team_name = {'ATL': 'Atlanta', 'BOS': 'Boston', 'BRK': 'Brooklyn', 'CHO': 'Charlotte', 'CHI': 'Chicago', 'CLE': 'Cleveland', 'DAL': 'Dallas', 'DEN': 'Denver', 'DET': 'Detroit', 'GSW': 'GoldenState', 'HOU': 'Houston', 'IND': 'Indiana', 'LAC': 'LAClippers', 'LAL': 'LALakers', 'MEM': 'Memphis', 'MIA': 'Miami', 'MIL': 'Milwaukee', 'MIN': 'Minnesota', 'NOP': 'NewOrleans', 'NYK': 'NewYork', 'OKC': 'OklahomaCity', 'ORL': 'Orlando', 'PHI': 'Philadelphia', 'PHO': 'Phoenix', 'POR': 'Portland', 'SAC': 'Sacramento', 'SAS': 'SanAntonio', 'TOR': 'Toronto', 'UTA': 'Utah', 'WAS': 'Washington'}
# team_abbr_to_sr_review_team_name = {'Atlanta Hawks': 'Atlanta', 'Boston Celtics': 'Boston', 'Brooklyn Nets': 'Brooklyn', 'Charlotte Hornets': 'Charlotte', 'Chicago Bulls': 'Chicago', 'Cleveland Cavaliers': 'Cleveland', 'Dallas Mavericks': 'Dallas', 'Denver Nuggets': 'Denver', 'Detroit Pistons': 'Detroit', 'Golden State Warriors': 'GoldenState', 'Houston Rockets': 'Houston', 'Indiana Pacers': 'Indiana', 'Los Angeles Clippers': 'LAClippers', 'Los Angeles Lakers': 'LALakers', 'Memphis Grizzlies': 'Memphis', 'Miami Heat': 'Miami', 'Milwaukee Bucks': 'Milwaukee', 'Minnesota Timberwolves': 'Minnesota', 'New Orleans Pelicans': 'NewOrleans', 'New York Knicks': 'New York', 'Oklahoma City Thunder': 'OklahomaCity', 'Orlando Magic': 'Orlando', 'Philadelphia 76ers': 'Philadelphia', 'Phoenix Suns': 'Phoenix', 'Portland Trail Blazers': 'Portland', 'Sacramento Kings': 'Sacramento', 'San Antonio Spurs': 'SanAntonio', 'Toronto Raptors': 'Toronto', 'Utah Jazz': 'Utah', 'Washington Wizards': 'Washington'}

pbp_dict = pickle.load(open('2023_archive.pickle', 'rb'))
# pbp_dict = {}
counter = 0
missing = []
for bs_id in boxscore_ids.keys():
    print(bs_id)
    home_abbr = boxscore_ids[bs_id]['home_abbr']
    away_abbr = boxscore_ids[bs_id]['away_abbr']
    new_bs_id = bs_id[:9] + sorted([team_abbr_to_sr_review_team_name[home_abbr], team_abbr_to_sr_review_team_name[away_abbr]])[0]
    if new_bs_id in pbp_dict.keys():
        continue
    
    if new_bs_id not in odds_dict.keys():
        missing.append(new_bs_id)
        print('missing', new_bs_id)
        continue
    pbp_df = get_pbp_df(bs_id, home_abbr, away_abbr)
    pbp_df['home_close_spread'] = odds_dict[new_bs_id]
    pbp_df = format_pbp(pbp_df)
    pbp_df = add_win_prob_xgb(pbp_df)
    pbp_df.iloc[-1, pbp_df.columns.get_loc('home_win_prob')] = 1 if pbp_df['home_margin'].iloc[-1] > 0 else 0
    pbp_df['home_team_name'] = pbp_df['home_name']
    pbp_df['away_team_name'] = pbp_df['away_name']
    fig = make_plot(pbp_df)
    excitement_index = get_excitement_index(pbp_df)
    dominance_index = get_dominance_index(pbp_df)
    tension_index = get_tension_index(pbp_df)

    pbp_dict[new_bs_id] = {'df': pbp_df, 'excitement_index': excitement_index, 'dominance_index': dominance_index, 'tension_index': tension_index, 'plot': fig}

    print(new_bs_id)
    counter += 1
    print(counter)
    # save to pickle
    pickle.dump(pbp_dict, open('2023_archive.pickle', 'wb'))

print('missing:', missing)


