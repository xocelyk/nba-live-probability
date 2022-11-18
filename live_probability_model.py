import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
# import random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# import grid search cv
from sklearn.model_selection import GridSearchCV
from random import random
from matplotlib import pyplot as plt
# import standard scaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import log_loss
import os
import random
import xgboost as xgb

'''
Next steps (not prioritized):
1. impute possession
2. factor in current event (free throws, etc.)
3. investigate momentum? autocorrelation?
4. smooth the weights
5. CV for max depth in DTree
6. smooth the DTree / logistic regression transition
7. cache models for quicker computer
8. eventually deploy model to intake live data
9. diff on the margins (check)
10. try using xgboost
'''

def time_elapsed(row):
    num_overtimes = row['period'] - 5
    if row['period'] < 5:
        return (row['period'] - 1) * 12 + 12 - row['time_in_period']
    else:
        return 48 + num_overtimes * 5 + 5 - row['time_in_period']


def get_models_dict(train_df, test_df, x_features, y_features):
    # concat
    models_dict = {'lr': {}}
    weights_dict = {}
    X_train, X_test, y_train, y_test = train_df[x_features], train_df[y_features], test_df[x_features], test_df[y_features]
    df = pd.concat([train_df, test_df])
    X = df[x_features]
    y = df[y_features]
    round_by = 0.2
    for time_remaining in np.arange(0, 48 + round_by, round_by):
        time_remaining = round(time_remaining, 1)
        print('time_remaining: ', time_remaining)
        x = np.array([time_remaining, 0, 0]).reshape(1, -1)
        tau = validate_weights(X_train, X_test, y_train, y_test, x, feature_name='time_remaining', round_by=round_by)
        model = LogisticRegression()
        weights = sample_weights(tau, df, x, feature_name='time_remaining')
        weights_dict[time_remaining] = weights
        model.fit(X, y.values.ravel(), sample_weight = weights)
        models_dict['lr'][time_remaining] = model
    dtree = DTree(X, y, max_depth=8)
    models_dict['dtree'] = dtree
    return models_dict

def get_xgboost_model(train_df, test_df, x_features, y_features):
    # TODO: this is terrible lol
    X_train, y_train, X_test, y_test = train_df[x_features], train_df[y_features], test_df[x_features], test_df[y_features]
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
    # K fold CV
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
                    'objective':['binary:logistic'],
                    'learning_rate': [0.05], #so called `eta` value
                    'max_depth': [6],
                    'min_child_weight': [11],
                    'silent': [1],
                    'subsample': [0.8],
                    'colsample_bytree': [0.7],
                    'n_estimators': [5], #number of trees, change it to 1000 for better results
                    'missing':[-999],
                    'seed': [1337]}
    clf = GridSearchCV(xgb_model, parameters, n_jobs=5,
                        cv=2,
                        scoring='neg_log_loss',
                        verbose=2, refit=True)
    
    clf.fit(X, y.values.ravel())
    print('best params: ', clf.best_params_)
    print('best score: ', clf.best_score_)
    return clf


def predict(row, models_dict):
    time_remaining = row['time_remaining']
    x = row.values.reshape(1, -1)
    dtree_threshold = 1
    closest_dist = float('inf')
    closest_time_remaining = None
    best_model = None
    for key, model in models_dict['lr'].items():
        dist = abs(time_remaining - key)
        if dist < closest_dist:
            closest_dist = dist
            closest_time_remaining = key
            best_model = model
    if time_remaining > dtree_threshold:
        # find closest time_remaining in models_dict.keys()
        print(x)
        print(model.predict_proba(x))
        return best_model.predict_proba(x)[0][1]
    else:
        return blend_preds(dtree_threshold, best_model, models_dict['dtree'], x, time_remaining)


def blend_preds(threshold, model, dtree, x, time_remaining):
    model_pred = model.predict_proba(x)[0][1]
    dtree_pred = dtree.predict_proba(x)[0][1]
    smooth_fn = lambda p1, p2, threshold, time_remaining: ((p1 * time_remaining) + p2 * (threshold - time_remaining))/(threshold)
    p1 = model_pred
    p2 = dtree_pred
    return smooth_fn(p1, p2, threshold, time_remaining)
    
def model(train_df, test_df, x_features, y_features, verbose=False):

    # This is what was passed
    # x_features = ['timeRemaining', 'homeMargin', 'home_close_spread']
    # y_features = ['home_win']

    X_train = train_df[x_features]
    y_train = train_df[y_features]
    X_test = test_df[x_features]
    y_test = test_df[y_features]

    time = []
    proba = []
    margin = []

    dtree_threshold = 1

    X_train_under_threshold = X_train[X_train['time_remaining'] < dtree_threshold]
    y_train_under_threshold = y_train[X_train['time_remaining'] < dtree_threshold]
    X_test_under_threshold = X_test[X_test['time_remaining'] < dtree_threshold]
    y_test_under_threshold = y_test[X_test['time_remaining'] < dtree_threshold]

    dtree = DTree(X_train_under_threshold, y_train_under_threshold)

    ### preload weights by 1 min intervals
    weights_dict = {}
    for timeRemaining in np.arange(0, 49, 1):
        x = np.array([timeRemaining, 0, 0, 0]).reshape(1, -1)
        weights = validate_weights(X_train, X_test, y_train, y_test, x, feature_name='time_remaining')
        weights_dict[timeRemaining] = weights

    for idx, row in X_test.iterrows():
        time_remaining = row['time_remaining']
        home_margin = row['home_margin']
        home_close_spread = row['home_close_spread']
        home_win = y_test.loc[idx, 'home_win']
        x = row[['time_remaining', 'home_margin', 'home_margin_diff', 'home_close_spread']]
        # x = np.array([time_remaining, home_margin, home_close_spread]).reshape(1, -1)

        if time_remaining > dtree_threshold:
            timeRemaining_int = int(round(time_remaining))
            weights = weights_dict[timeRemaining_int]
            model = LogisticRegression()
            model.fit(X_train.values, y_train.values.ravel(), sample_weight = weights)

            pred = model.predict(x)[0]
            pred_proba = model.predict_proba(x)[0][1]
            if verbose:
                print(pd.DataFrame({'timeRemaining': [time_remaining], 'margin': [home_margin], 'home_close_spread': [home_close_spread], 'home_win': [home_win], 'pred': [pred], 'pred_proba': [pred_proba]}))
        else:
            pred = dtree.predict(x)[0]
            pred_proba = dtree.predict_proba(x)[0][1]
            if verbose:
                print(pd.DataFrame({'timeRemaining': [time_remaining], 'margin': [home_margin], 'home_close_spread': [home_close_spread], 'home_win': [home_win], 'pred': [pred], 'pred_proba': [pred_proba]}))

        if time_remaining == 0:
            plt.plot(time[::-1], proba)
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.show()
            time = []
            proba = []
            margin = []
        else:
            time.append(time_remaining)
            proba.append(pred_proba)
            margin.append(home_margin)

def sample_weights(tau, df, x, feature_name):
    domain = df[[feature_name]]
    x0 = x[:, df.columns.get_loc(feature_name)]
    weights = np.exp(-np.linalg.norm(domain - x0, axis = 1) / (2 * tau ** 2))
    return weights


def validate_weights(x_train, x_test, y_train, y_test, x, feature_name, round_by = 1):
    # TODO: getting different results for similar times, smooth somehow
    # TODO: cross walidate tau
    # we should see the band width of the weights decrease as the game progresses
    # domain = x_train[[feature_name]]
    # x0 = x[:, x_train.columns.get_loc(feature_name)]
    # # print('time:', x0)
    # error_lst = []
    # tau_lst = np.logspace(-1, 1, 10)
    # best_tau = None
    # min_error = float('inf')
    # best_weights = None
    # x_test_time_slice = x_test[abs(x_test[feature_name] - x0) < round_by / 2]
    # y_test_time_slice = y_test[abs(x_test[feature_name] - x0) < round_by / 2]
    # x_train_time_slice = x_train[abs(x_train[feature_name] - x0) < round_by / 2]
    # y_train_time_slice = y_train[abs(x_train[feature_name] - x0) < round_by / 2]
    # for tau in tau_lst:
    #     weights = np.exp(-np.linalg.norm(domain - x0, axis = 1) / (2 * tau ** 2))
    #     model = LogisticRegression().fit(x_train, y_train.values.ravel(), sample_weight=weights)
    #     pred_proba = model.predict_proba(x_test_time_slice)
    #     error = log_loss(y_test_time_slice.values.ravel(), pred_proba)
    #     error_lst.append(error)
    #     if error <= min_error:
    #         min_error = error
    #         best_weights = weights
    #         best_tau = tau
    
    tau = .5
    best_tau = .5
    return best_tau

def DTree(x_train, y_train, max_depth=8, time_remaining_threshold=1):
    # use when timeRemaining < 1 min
    from sklearn import tree
    from sklearn.tree import export_text
    # TODO: this is broken
    # x_train = x_train[x_train['time_remaining'] < time_remaining_threshold]
    # y_train = y_train.loc[x_train.index]
    #TODO: fine tune this, 7 is probably a little too deep
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf = clf.fit(x_train, y_train)
    r = export_text(clf, feature_names=list(x_train.columns))
    print(r)
    return clf
    
def time_remaining(row):
    if row['time_elapsed'] <= 48:
        return 48 - row['time_elapsed']
    else:
        return 5 - ((row['time_elapsed'] - 48) % 5)

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

def format_historic_df(df):
    pbp_dict = {}
    df['home_margin'] = df['HomeScore'] - df['AwayScore']
    df['home_win'] = int(df.iloc[-1]['home_margin'] > 0)
    df['time_in_period'] = df.apply(lambda row: float(str(row['Time'].split(':')[0])) + float(str(row['Time'].split(':')[1]))/60, axis=1)
    df.rename(columns={'HomeScore': 'home_score', 'AwayScore': 'away_score', 'Period': 'period', 'HomeName': 'home_name', 'AwayName': 'away_name'}, inplace=True)
    df = df[['period', 'time_in_period', 'home_name', 'away_name', 'home_score', 'away_score', 'home_margin', 'home_close_spread','home_win']]
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
            df = pd.read_csv('pbp_data/with_odds/' + subdir + '/' + file)
            df = format_historic_df(df)
            pbp_dict[game_id] = df
    return pbp_dict

def train_test_validate_split_games(df_dict, test_size=0.2, validate_size=0.2):
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
        pbp_dict = pickle.load(open('training_data/pbp_data.pickle', 'rb'))
        train_df, test_df, validate_df = train_test_validate_split_games(pbp_dict, test_size=.2)
        train_df['home_margin_diff'] = train_df['home_margin'].diff()
        train_df['home_margin_diff_2'] = train_df['home_margin_diff'].diff()
        train_df['home_margin_diff'].fillna(0, inplace=True)
        train_df['home_margin_diff_2'].fillna(0, inplace=True)
        test_df['home_margin_diff'] = test_df['home_margin'].diff()
        test_df['home_margin_diff_2'] = test_df['home_margin_diff'].diff()
        test_df['home_margin_diff'].fillna(0, inplace=True)
        test_df['home_margin_diff_2'].fillna(0, inplace=True)
        validate_df['home_margin_diff'] = validate_df['home_margin'].diff()
        validate_df['home_margin_diff_2'] = validate_df['home_margin_diff'].diff()
        validate_df['home_margin_diff'].fillna(0, inplace=True)
        validate_df['home_margin_diff_2'].fillna(0, inplace=True)
        # train_df = pickle.load(open('training_data/train_df.pickle', 'rb'))
        # test_df = pickle.load(open('training_data/test_df.pickle', 'rb'))
    else:
        pbp_dict_2023 = load_2023_data()
        pbp_dict_historic = load_historic_data()
        # merge dicts
        pbp_dict = {**pbp_dict_2023, **pbp_dict_historic}
        pickle.dump(pbp_dict, open('training_data/pbp_data.pickle', 'wb'))
        train_df, test_df, validate_df = train_test_validate_split_games(pbp_dict, test_size=.2)
        pickle.dump(train_df, open('training_data/train_df.pickle', 'wb'))
        pickle.dump(test_df, open('training_data/test_df.pickle', 'wb'))
        pickle.dump(validate_df, open('training_data/validate_df.pickle', 'wb'))

    train_df['time_elapsed'] = train_df.apply(time_elapsed, axis=1)
    train_df['time_remaining'] = train_df.apply(time_remaining, axis=1)
    test_df['time_elapsed'] = test_df.apply(time_elapsed, axis=1)
    test_df['time_remaining'] = test_df.apply(time_remaining, axis=1)
    validate_df['time_elapsed'] = validate_df.apply(time_elapsed, axis=1)
    validate_df['time_remaining'] = validate_df.apply(time_remaining, axis=1)

    x_features = ['time_remaining', 'home_margin', 'home_margin_diff', 'home_close_spread']
    y_features = ['home_win']

    all_features = x_features + y_features
    train_df = train_df[all_features]
    test_df = test_df[all_features]
    validate_df = validate_df[all_features]

    train_df = train_df.dropna()
    test_df = test_df.dropna()
    validate_df = validate_df.dropna()
    train_df = train_df.drop_duplicates(keep='first')
    test_df = test_df.drop_duplicates(keep='first')
    validate_df = validate_df.drop_duplicates(keep='first')
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
    return train_df, test_df, validate_df, x_features, y_features

def test_lr_model(train_df, test_df, validation_df, x_features, y_features):
    models_dict = get_models_dict(train_df, test_df, x_features, y_features)
    pickle.dump(models_dict, open('training_data/models_dict.pickle', 'wb'))
    preds = []
    res_data = []
    for idx, row in validation_df.iterrows():
        x = row[x_features]
        proba = predict(x, models_dict)
        preds.append(proba)
        # add to validation_df
        res_data.append(x.values.tolist() + [proba] + [row['home_win']])
    res_df = pd.DataFrame(res_data)
    res_df.to_csv('pred_results.csv')
    print(log_loss(validation_df['home_win'], preds))

def test_xgboost_model(train_df, test_df, validation_df, x_features, y_features):
    model = get_xgboost_model(train_df, test_df, x_features, y_features)
    dtree = DTree(validation_df[x_features], validation_df[y_features], time_remaining_threshold=1)
    pickle.dump(model, open('training_data/xgboost_model.pickle', 'wb'))
    preds = []
    res_data = []
    for idx, row in validation_df.iterrows():
        proba = predict_xgboost(row[x_features], model, dtree)
        preds.append(proba)
        # print([pd.DataFrame(row[x_features]).T, proba, row['home_win']])
    print(log_loss(validation_df['home_win'], preds))

def predict_xgboost(row, model, dtree, use_dtree=True):
    time_remaining = row['time_remaining']
    x = row.values.reshape(1, -1)
    dtree_threshold = 1
    if not use_dtree:
        return model.predict_proba(x)[0][1]
    else:
        if time_remaining > dtree_threshold:
            # find closest time_remaining in models_dict.keys()
            return model.predict_proba(x)[0][1]
        else:
            return blend_preds(dtree_threshold, model, dtree, x, time_remaining)


def playgrond():
    pred_df = pd.read_csv('pred_results.csv')
    pred_df.columns = ['id', 'time_remaining', 'home_margin', 'home_margin_diff', 'home_close_spread', 'proba', 'home_win']
    pred_df['time_remaining_int'] = round(pred_df['time_remaining'], 0).astype(int)
    for time in pred_df['time_remaining_int'].unique():
        error = log_loss(pred_df[pred_df['time_remaining_int'] == time]['home_win'], pred_df[pred_df['time_remaining_int'] == time]['proba'])
        print('Time Remaining: {}, Log Loss: {}'.format(time, error))

def main():
    import warnings
    warnings.filterwarnings("ignore")
    train_df, test_df, validation_df, x_features, y_features = get_data(preload=True)
    df = pd.concat([train_df, test_df, validation_df])
    test_lr_model(train_df, test_df, validation_df, x_features, y_features)
    # test_xgboost_model(train_df, test_df, validation_df, x_features, y_features)


if __name__ == '__main__':
    main()

