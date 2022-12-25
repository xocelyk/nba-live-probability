import pickle
import pandas as pd

def get_pbp_dict():
    filename = '2023_archive.pickle'
    with open(filename, 'rb') as f:
        pbp_dict = pickle.load(f)
    return pbp_dict


def get_pbp_df(pbp_dict):
    pbp_df = pd.DataFrame(columns=['gameId', 'Date', 'Home', 'Away', 'HomeScore', 'AwayScore', 'Tension', 'Excitement', 'Dominance'])

    for boxscore_id, dct in pbp_dict.items():
        year = boxscore_id[:4]
        month = boxscore_id[4:6]
        day = boxscore_id[6:8]
        date = '{}-{}-{}'.format(year, month, day)
        home_name = dct['df']['home_name'].iloc[0]
        away_name = dct['df']['away_name'].iloc[0]
        home_score = dct['df']['home_score'].iloc[-1]
        away_score = dct['df']['away_score'].iloc[-1]
        tension = dct['tension_index']
        excitement = dct['excitement_index']
        dominance = dct['dominance_index']
        gameId = boxscore_id

        to_add = pd.DataFrame.from_dict({'gameId': gameId, 'Date': date, 'Home': home_name, 'Away': away_name, 'HomeScore': home_score, 'AwayScore': away_score, 'Tension': tension, 'Excitement': excitement, 'Dominance': dominance}, orient='index').T
        pbp_df = pd.concat([pbp_df, to_add], ignore_index=True)

    pbp_df.drop('gameId', axis=1, inplace=True)
    return pbp_df


def get_dominance_rankings():
    pbp_dict = get_pbp_dict()
    dominance_scores = {}
    for boxscore_id, dct in pbp_dict.items():
        home = dct['df']['home_name'].iloc[0]
        away = dct['df']['away_name'].iloc[0]
        home_dominance = dct['dominance_index']
        away_dominance = 100 - home_dominance
        if home not in dominance_scores:
            dominance_scores[home] = [home_dominance]
        else:
            dominance_scores[home].append(home_dominance)
        if away not in dominance_scores:
            dominance_scores[away] = [away_dominance]
        else:
            dominance_scores[away].append(away_dominance)

    dominance_df = pd.DataFrame(columns=['Team', 'Dominance'])
    for team, scores in dominance_scores.items():
        df_to_add = pd.DataFrame.from_dict({'Team': team, 'Dominance': sum(scores) / len(scores)}, orient='index').T
        dominance_df = pd.concat([dominance_df, df_to_add], ignore_index=True)

    dominance_df.sort_values('Dominance', ascending=False, inplace=True)
    dominance_df['Rank'] = dominance_df['Dominance'].rank(ascending=False).astype(int)
    dominance_df.sort_values('Rank', inplace=True)
    dominance_df.set_index('Rank', inplace=True)
    return dominance_df

def main():
    dominance_df = get_dominance_rankings()
    dominance_df['Rank'] = dominance_df['Dominance'].rank(ascending=False).astype(int)
    dominance_df.sort_values('Rank', inplace=True)
    dominance_df.set_index('Rank', inplace=True)
    print(dominance_df)
    dominance_df.to_csv('model_results/dominance_rankings.csv', index=False)


