import pandas as pd
import numpy as np
import os
import shutil

pbp_dir = '/Users/kylecox/Documents/ws/nba-pbp/pbp_in_out'
pbp_names = []
pbp_files = os.listdir(pbp_dir)
for filename in pbp_files:
    pbp_names.append(filename[9:12])
pbp_names = list(set(pbp_names))

odds_names = []
odds_dir = '/Users/kylecox/Documents/ws/nba/live_probability/odds_data/formatted'
odds_files = os.listdir(odds_dir)
for filename in odds_files: 
    # ignore hidden files
    if filename[0] == '.':
        continue
    print(filename)
    full_path = os.path.join(odds_dir, filename)
    df = pd.read_csv(full_path)
    print(df.tail())
    odds_names += df['home_team'].unique().tolist()
odds_names = list(set(odds_names))

pbp_names_to_odds_names = {}

# ['OklahomaCity', 'Portland', 'Charlotte', 'Houston', 'LAClippers', 'Toronto', 'Memphis', 'Atlanta', 'Detroit', 'Cleveland', 'NewYork', 'Philadelphia', 'Washington', 'GoldenState', 'Golden State', 'Dallas', 'NewJersey', 'NewOrleans', 'Phoenix', 'Milwaukee', 'Utah', 'Sacramento', 'Orlando', 'Miami', 'Brooklyn', 'Chicago', 'Denver', 'SanAntonio', 'Boston', 'Indiana', 'Minnesota', 'LALakers']
# ['ORL', 'TOR', 'ATL', 'LAL', 'DET', 'CHA', 'NJN', 'MIA', 'LAC', 'MIL', 'CHH', 'IND', 'SEA', 'NYK', 'POR', 'MIN', 'SAC', 'MEM', 'NOP', 'DAL', 'CHO', 'GSW', 'BRK', 'BOS', 'NOK', 'OKC', 'UTA', 'WAS', 'HOU', 'DEN', 'PHI', 'CLE', 'SAS', 'CHI', 'PHO', 'NOH', 'VAN']

pbp_names_to_odds_names['ORL'] = 'Orlando'
pbp_names_to_odds_names['TOR'] = 'Toronto'
pbp_names_to_odds_names['ATL'] = 'Atlanta'
pbp_names_to_odds_names['LAL'] = 'LALakers'
pbp_names_to_odds_names['DET'] = 'Detroit'
pbp_names_to_odds_names['CHA'] = 'Charlotte'
pbp_names_to_odds_names['NJN'] = 'NewJersey'
pbp_names_to_odds_names['MIA'] = 'Miami'
pbp_names_to_odds_names['LAC'] = 'LAClippers'
pbp_names_to_odds_names['MIL'] = 'Milwaukee'
pbp_names_to_odds_names['CHH'] = 'Charlotte'
pbp_names_to_odds_names['IND'] = 'Indiana'
pbp_names_to_odds_names['SEA'] = 'Seattle'
pbp_names_to_odds_names['NYK'] = 'NewYork'
pbp_names_to_odds_names['POR'] = 'Portland'
pbp_names_to_odds_names['MIN'] = 'Minnesota'
pbp_names_to_odds_names['SAC'] = 'Sacramento'
pbp_names_to_odds_names['MEM'] = 'Memphis'
pbp_names_to_odds_names['NOP'] = 'NewOrleans'
pbp_names_to_odds_names['DAL'] = 'Dallas'
pbp_names_to_odds_names['CHO'] = 'Charlotte'
pbp_names_to_odds_names['GSW'] = 'GoldenState'
pbp_names_to_odds_names['BRK'] = 'Brooklyn'
pbp_names_to_odds_names['BOS'] = 'Boston'
pbp_names_to_odds_names['NOK'] = 'NewOrleans'
pbp_names_to_odds_names['OKC'] = 'OklahomaCity'
pbp_names_to_odds_names['UTA'] = 'Utah'
pbp_names_to_odds_names['WAS'] = 'Washington'
pbp_names_to_odds_names['HOU'] = 'Houston'
pbp_names_to_odds_names['DEN'] = 'Denver'
pbp_names_to_odds_names['PHI'] = 'Philadelphia'
pbp_names_to_odds_names['CLE'] = 'Cleveland'
pbp_names_to_odds_names['SAS'] = 'SanAntonio'
pbp_names_to_odds_names['CHI'] = 'Chicago'
pbp_names_to_odds_names['PHO'] = 'Phoenix'
pbp_names_to_odds_names['NOH'] = 'NewOrleans'
pbp_names_to_odds_names['VAN'] = 'Vancouver'


odds_names_to_pbp_names = {}
for key, value in pbp_names_to_odds_names.items():
    if value not in odds_names_to_pbp_names:
        odds_names_to_pbp_names[value] = [key]
    else:
        odds_names_to_pbp_names[value].append(key)

# {'OklahomaCity': ['OKC'], 'Portland': ['POR'], 'Charlotte': ['CHA', 'CHO'], 'Houston': ['HOU'], 'LAClippers': ['LAC'], 'Toronto': ['TOR'], 'Memphis': ['MEM'], 'Atlanta': ['ATL'], 'Detroit': ['DET'], 'Cleveland': ['CLE'], 'NewYork': ['NYK'], 'Philadelphia': ['PHI'], 'Washington': ['WAS'], 'GoldenState': ['GSW'], 'Golden State': ['GSW'], 'Dallas': ['DAL'], 'NewJersey': ['NJN'], 'NewOrleans': ['NOP', 'NOK', 'NOH'], 'Phoenix': ['PHO'], 'Milwaukee': ['MIL'], 'Utah': ['UTA'], 'Sacramento': ['SAC'], 'Orlando': ['ORL'], 'Miami': ['MIA'], 'Brooklyn': ['BRK'], 'Chicago': ['CHI'], 'Denver': ['DEN'], 'SanAntonio': ['SAS'], 'Boston': ['BOS'], 'Indiana': ['IND'], 'Minnesota': ['MIN'], 'LALakers': ['LAL']}

update_pbp = True

if update_pbp:
    pbp_dir = '/Users/kylecox/Documents/ws/nba-pbp/pbp_in_out'
    for filename in pbp_files:
        fullpath = os.path.join(pbp_dir, filename)
        team_abbr = filename[9:12]
        if team_abbr in pbp_names_to_odds_names:
            new_filename = filename.replace(team_abbr, pbp_names_to_odds_names[team_abbr])
            year = int(filename[:4])
            month = int(filename[4:6])
            if 8 < month <= 12:
                season = year + 1
            else:
                season = year
            new_dir = 'pbp_data/no_odds/' + str(season) + '/'
            new_path = new_dir + new_filename
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            # copy old file to new path
            df = pd.read_csv(fullpath)
            df.to_csv(new_path, index=False)
        else:
            print('Missing:', team_abbr)


odds_by_year = {}
odds_dir = '/Users/kylecox/Documents/ws/nba/live_probability/odds_data/formatted'
for filename in odds_files:
    if filename.startswith('.'):
        continue
    fullpath = os.path.join(odds_dir, filename)
    print(fullpath)
    year_df = pd.read_csv(fullpath)
    odds_by_year[filename[:4]] = year_df

#20151027Chicago

# print(odds_by_year.keys())
pbp_dir = '/Users/kylecox/Documents/ws/nba/live_probability/pbp_data/no_odds'
pbp_subdirs = [d for d in os.listdir(pbp_dir) if os.path.isdir(os.path.join(pbp_dir, d))]
for subdir in pbp_subdirs:
    if subdir not in odds_by_year:
        print('Missing Odds for Year:', subdir)
        continue
    odds_df = odds_by_year[subdir]
    odds_df.set_index('game_id', inplace=True)
    new_dir = 'pbp_data/with_odds/' + subdir + '/'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for file in os.listdir(os.path.join(pbp_dir, subdir)):
        game_id = file[:-4]
        fullpath = os.path.join(pbp_dir, subdir, file)
        if game_id in odds_df.index:
            newpath = new_dir + file
            pbp_df = pd.read_csv(fullpath)
            try:
                pbp_df['home_open_spread'] = odds_df.loc[game_id]['home_open_spread']
                pbp_df['home_close_spread'] = odds_df.loc[game_id]['home_close_spread']
            except:
                print('Error:', game_id)
                continue
            pbp_df.drop([col for col in pbp_df.columns if col.startswith('Unnamed')], axis=1, inplace=True)
            if set(['ActivePlayers', 'A1', 'A2', 'A3', 'A4', 'A5', 'H1', 'H2', 'H3', 'H4', 'H5']).issubset(pbp_df.columns):
                pbp_df.drop(['ActivePlayers', 'A1', 'A2', 'A3', 'A4', 'A5', 'H1', 'H2', 'H3', 'H4', 'H5'], axis=1, inplace=True)
            print(newpath)
            pbp_df.to_csv(newpath, index=False)
    # print(odds_df.index)



