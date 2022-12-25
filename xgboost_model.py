import pickle
import pandas as pd
import numpy as np
import random
import os
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import log_loss
import time

import warnings
warnings.filterwarnings('ignore')

ALL_WORDS = []

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

def cval_xgboost(X, y):
    df = pd.concat([X, y], axis=1)

    def train_test_split_by_game(df, test_size=0.2):
        # we can't use regular train_test_split because the model can learn the outcomes of games in the test set
        # have to split by entire games instead
        games = list(df.index.unique())
        random.shuffle(games)
        test_games = games[:int(len(games) * test_size)]
        train_games = games[int(len(games) * test_size):]
        train_df = df[df.index.isin(train_games)]
        test_df = df[df.index.isin(test_games)]
        return train_df, test_df

    train_df, test_df = train_test_split_by_game(df)
    X_train = train_df.drop(['home_win'], axis=1)
    y_train = train_df['home_win']
    X_test = test_df.drop(['home_win'], axis=1)
    y_test = test_df['home_win']
    params = {
        'gamma': [1, 2, 5],
        'subsample': [0.6, 1.0],
        'max_depth': [3, 5, 7, 9]
    }

    xgb_model = xgb.XGBClassifier()
    clf = GridSearchCV(xgb_model, params, n_jobs=1,
                          scoring='neg_log_loss',
                            cv=StratifiedKFold(n_splits=5, shuffle=True),
                            verbose=3, refit=True)
    clf.fit(X_train, y_train)
    print(clf.best_score_)
    print(clf.best_params_)
    y_pred = clf.predict(X_test)
    # describe predictions
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return clf



def train_xgboost(df, x_features, y_features):
    model = xgb.XGBClassifier(gamma=1, max_depth=7, min_child_weight=1, subsample=0.6, colsample_bytree=1.0)
    X = df[x_features]
    y = df[y_features]
    model.fit(X, y)
    # make predictions for test data
    y_pred = model.predict(X)
    predictions = [round(value) for value in y_pred]
    pred_proba = model.predict_proba(X)
    plt.hist(pred_proba[:,1])
    plt.show()
    print(pd.DataFrame(pred_proba).describe())
    # evaluate predictions
    accuracy = accuracy_score(y, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(classification_report(y, predictions))
    # print log loss
    print('Log Loss: {}'.format(log_loss(y, pred_proba)))

    # print feature importance by feature name
    print('Feature Importance')
    print(model.feature_importances_)
    print([x for x in zip(x_features, model.feature_importances_)])
    return model


def format_historic_df(df):
    df['home_margin'] = df['HomeScore'] - df['AwayScore']
    df['home_win'] = int(df.iloc[-1]['home_margin'] > 0)
    df['time_in_period'] = df.apply(lambda row: float(str(row['Time'].split(':')[0])) + float(str(row['Time'].split(':')[1]))/60, axis=1)
    df['time_elapsed'] = df.apply(lambda row: time_elapsed(row), axis=1)
    df['time_remaining'] = df.apply(lambda row: time_remaining(row), axis=1)
    try:
        df = add_possessions(df)
        df = add_possession_by_team(df)
    except:
        print('Error adding possessions')
        print(df.head())
        return None
    print([col for col in df.columns])
    df['HomePossession'] = df['HomePossession'].astype(int)
    df.rename(columns={'HomeScore': 'home_score', 'AwayScore': 'away_score', 'Period': 'period', 'HomeName': 'home_name', 'AwayName': 'away_name', 'HomePossession': 'home_possession'}, inplace=True)
    df.to_csv('test.csv')
    df = df[['period', 'time_elapsed', 'time_remaining', 'time_in_period', 'home_name', 'away_name', 'home_score', 'away_score', 'home_possession', 'home_margin', 'home_close_spread','home_win', 'HomeEvent', 'AwayEvent']]
    return df


def load_historic_data():
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
            df = format_historic_df(df)
            if df is None:
                continue
            df['game_id'] = game_id
            pbp_dict[game_id] = df
    return pbp_dict


def load_2023_data():
    pbp_dict = pickle.load(open('nba_dot_com_data/pbp_data_2023.pickle', 'rb'))
    for game_id, df in pbp_dict.items():
        df.rename(columns={'timeInPeriod': 'time_in_period', 'scoreHome': 'home_score', 'scoreAway': 'away_score'}, inplace=True)
        df['home_score'] = df['home_score'].astype(int)
        df['away_score'] = df['away_score'].astype(int)
        df['home_margin'] = df['home_score'] - df['away_score']
        df['home_name'] = df['hTeam_city'] + ' '  + df['hTeam_name']
        df['away_name'] = df['aTeam_city'] + ' '  + df['aTeam_name']
        df = df[['period', 'time_in_period', 'home_name', 'away_name', 'home_score', 'away_score', 'home_margin', 'home_close_spread', 'home_win']]
        pbp_dict[game_id] = df
    return pbp_dict

def train_test_validate_split_games(df_dict, test_size=0.2, validate_size=0.1):
    # we can't use regular train_test_split because the model can learn the outcomes of games in the test set
    # have to split by entire games instead
    train_dict = {}
    test_dict = {}
    validate_dict = {}
    games = list(df_dict.keys())
    random.shuffle(games)
    test_games = games[:int(len(games) * test_size)]
    validate_games = games[int(len(games) * test_size):int(len(games) * (test_size + validate_size))]
    train_games = games[int(len(games) * (test_size + validate_size)):]
    # concat all test_games
    test_df = pd.concat([df_dict[game_id] for game_id in test_games])
    # concat all train_games
    train_df = pd.concat([df_dict[game_id] for game_id in train_games])
    # concat all validate_games
    validate_df = pd.concat([df_dict[game_id] for game_id in validate_games])
    return train_df, test_df, validate_df

def get_data(preload=True):
    if preload:
        pbp_dict = pickle.load(open('training_data/pbp_data_for_xgb.pickle', 'rb'))
    else:
        pbp_dict = load_historic_data()
        pickle.dump(pbp_dict, open('training_data/pbp_data_for_xgb.pickle', 'wb'))
    data = []
    for game_id, df in pbp_dict.items():
        data.append(df)
    df = pd.concat(data)
    df['home_margin_diff'] = df['home_margin'].diff()
    print(['df shape', df.shape])
    df.dropna(inplace=True)
    print(['df shape after dropna', df.shape])
    df.drop_duplicates(inplace=True)
    print(['df shape after drop_duplicates', df.shape])
    return df

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

def experimental_format(pbp_df):
    pbp_df['fts_remaining'] = pbp_df.apply(add_fts, axis=1)
    pbp_df['foul'] = pbp_df.apply(lambda row: 'foul' in row['AwayEvent'].lower() or 'foul' in row['HomeEvent'].lower(), axis=1)
    pbp_df['turnover'] = pbp_df.apply(lambda row: 'turnover' in row['AwayEvent'].lower() or 'turnover' in row['HomeEvent'].lower(), axis=1)
    pbp_df['steal'] = pbp_df.apply(lambda row: 'steal' in row['AwayEvent'].lower() or 'steal' in row['HomeEvent'].lower(), axis=1)
    pbp_df['block'] = pbp_df.apply(lambda row: 'block' in row['AwayEvent'].lower() or 'block' in row['HomeEvent'].lower(), axis=1)
    pbp_df['timeout'] = pbp_df.apply(lambda row: 'timeout' in row['AwayEvent'].lower() or 'timeout' in row['HomeEvent'].lower(), axis=1)
    pbp_df['offensive_foul'] = pbp_df.apply(lambda row: 'offensive foul' in row['AwayEvent'].lower() or 'offensive foul' in row['HomeEvent'].lower(), axis=1)
    pbp_df['defensive_foul'] = pbp_df.apply(lambda row: 'defensive foul' in row['AwayEvent'].lower() or 'defensive foul' in row['HomeEvent'].lower(), axis=1)
    pbp_df['offensive_rebound'] = pbp_df.apply(lambda row: 'offensive rebound' in row['AwayEvent'].lower() or 'offensive rebound' in row['HomeEvent'].lower(), axis=1)
    pbp_df['defensive_rebound'] = pbp_df.apply(lambda row: 'defensive rebound' in row['AwayEvent'].lower() or 'defensive rebound' in row['HomeEvent'].lower(), axis=1)
    return pbp_df

def parse_words(pbp_df):
    home_events = pbp_df['HomeEvent'].str.lower().str.split(' ')
    away_events = pbp_df['AwayEvent'].str.lower().str.split(' ')

    home_words = []
    away_words = []

    for event in home_events:
        home_words += event
    for event in away_events:
        away_words += event
    
    return home_words, away_words

def main():
    df = get_data(preload=True)
    # df = df.sample(frac=.1)
    df = experimental_format(df)


    df.set_index('game_id', inplace=True)
    # home_words, away_words = parse_words(df)
    # # get home words value counts
    # home_words = pd.Series(home_words).value_counts()
    # # to csv
    # home_words.to_csv('model_results/home_words.csv')
    # # get away words value counts
    # away_words = pd.Series(away_words).value_counts()
    # # to csv
    # away_words.to_csv('model_results/away_words.csv')

    # print(home_words)
    # print(away_words)


    x_features = ['time_remaining', 'home_margin', 'home_possession', 'home_close_spread', 'fts_remaining', 'foul', 'turnover', 'steal', 'block', 'timeout', 'offensive_foul', 'defensive_foul', 'offensive_rebound', 'defensive_rebound']
    # x_features = ['time_remaining', 'home_margin', 'home_close_spread']
    y_features = ['home_win']
    all_features = x_features + y_features
    df = df[all_features]
    X = df[x_features]
    y = df[y_features]

    # cval_xgboost(X, y)
    model = train_xgboost(df, x_features, y_features)
    # save pickle
    filename = 'model_results/xgboost_model.pickle'
    pickle.dump(model, open(filename, 'wb'))

if __name__ == '__main__':
    main()