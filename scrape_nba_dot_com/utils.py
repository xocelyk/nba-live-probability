import requests
import numpy as np
import pandas as pd

def request_pbp(game_id):
    url = 'https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_' + str(game_id) + '.json'
    r = requests.get(url)
    try:
        game_data = r.json()['game']['actions']
        return game_data
    except:
        return None

def scrape_today_odds():
    url = 'https://cdn.nba.com/static/json/liveData/odds/odds_todaysGames.json'
    odds_dict = {}
    r = requests.get(url)
    # try:
    games = r.json()['games']
    for game_data in games:
        game_id = game_data['gameId']
        home_team_id = game_data['homeTeamId']
        away_team_id = game_data['awayTeamId']
        odds_dict[game_id] = parse_today_odds(game_data)
    # except:
    #     return None
    return odds_dict

def parse_today_odds(game_data):
    # there is more info about movement if I would like to grab that
    game_odds = {}
    moneyline_books = game_data['markets'][0]['books']
    spread_books = game_data['markets'][1]['books']

    # get moneyline odds
    outcomes = None
    for book in moneyline_books:
        if 'outcomes' in book:
            outcomes = book['outcomes']
            break
    if outcomes is None:
        return None
    for market in outcomes:
        if market['odds_field_id'] == 1:
            # for home team
            assert market['type'] == 'home'
            game_odds['home_ml'] = market['odds']
        elif market['odds_field_id'] == 2:
            # for away team
            assert market['type'] == 'away'
            game_odds['away_ml'] = market['odds']

    # get spread odds
    outcomes = None
    for book in spread_books:
        if 'outcomes' in book:
            outcomes = book['outcomes']
            break
    if outcomes is None:
        return None
    for market in outcomes:
        if market['odds_field_id'] == 10:
            # for home team
            assert market['type'] == 'home'
            game_odds['home_spread'] = market['spread']
            game_odds['home_spread_odds'] = market['odds']
        elif market['odds_field_id'] == 12:
            # for away team
            assert market['type'] == 'away'
            game_odds['away_spread'] = market['spread']
            game_odds['away_spread_odds'] = market['odds']

    return game_odds

def parse_game(schedule, game_id):
    game_data = request_pbp(game_id)
    if game_data is None:
        return None
    game_pbp = {}
    schedule['game_id'] = schedule.index
    # home_open_spread = schedule[schedule['game_id'] == game_id]['home_open_spread'].tolist()[0]
    # home_close_spread = schedule[schedule['game_id'] == game_id]['home_close_spread'].tolist()[0]
    home_team_id = schedule[schedule['game_id'] == game_id]['hTeam_id'].tolist()[0]
    away_team_id = schedule[schedule['game_id'] == game_id]['aTeam_id'].tolist()[0]
    home_team_name = schedule[schedule['game_id'] == game_id]['hTeam_name'].tolist()[0]
    away_team_name = schedule[schedule['game_id'] == game_id]['aTeam_name'].tolist()[0]
    home_team_city = schedule[schedule['game_id'] == game_id]['hTeam_city'].tolist()[0]
    away_team_city = schedule[schedule['game_id'] == game_id]['aTeam_city'].tolist()[0]
    for action in game_data:
        actionNumber = action['actionNumber']
        clock = action['clock']
        period = action['period']
        periodType = action['periodType']
        actionType = action['actionType']
        subType = action['subType']
        qualifiers = action['qualifiers']
        personId = action['personId']
        x = action['x']
        y = action['y']
        side = action['side']
        shotDistance = action['shotDistance'] if 'shotDistance' in action else None
        shotResult = action['shotResult'] if 'shotResult' in action else None
        shotActionNumber = action['shotActionNumber'] if 'shotActionNumber' in action else None
        possession = action['possession']
        scoreHome = action['scoreHome']
        scoreAway = action['scoreAway']
        xLegacy = action['xLegacy']
        yLegacy = action['yLegacy']
        isFieldGoal = action['isFieldGoal']
        description = action['description']
        game_pbp[actionNumber] = [clock, period, periodType, actionType, subType, qualifiers, personId, x, y, side, shotDistance, shotResult, possession, scoreHome, scoreAway, xLegacy, yLegacy, isFieldGoal, description, home_team_id, away_team_id]
        if description == 'Game End':
            break
    game_pbp = pd.DataFrame.from_dict(game_pbp, orient='index', columns=['clock', 'period', 'periodType', 'actionType', 'subType', 'qualifiers', 'personId', 'x', 'y', 'side', 'shotDistance', 'shotResult', 'possession', 'scoreHome', 'scoreAway', 'xLegacy', 'yLegacy', 'isFieldGoal', 'description', 'home_team_id', 'away_team_id'])
    home_win = scoreHome > scoreAway
    game_pbp['home_win'] = int(home_win)
    # game_pbp['home_open_spread'] = home_open_spread
    # game_pbp['home_close_spread'] = home_close_spread
    game_pbp['hTeam_id'] = home_team_id
    game_pbp['aTeam_id'] = away_team_id
    game_pbp['hTeam_possession'] = game_pbp['possession'] == home_team_id
    game_pbp['aTeam_possession'] = game_pbp['possession'] == away_team_id
    game_pbp['timeMinutes'] = game_pbp['clock'].apply(lambda time: float(time[time.find('PT') + 2:time.find('M')]))
    game_pbp['timeSeconds'] = game_pbp['clock'].apply(lambda time: float(time[time.find('M')+1:time.find('S')]))
    game_pbp['timeInPeriod'] = game_pbp['timeMinutes'] + game_pbp['timeSeconds'] / 60
    game_pbp['hTeam_name'] = home_team_name
    game_pbp['aTeam_name'] = away_team_name
    game_pbp['hTeam_city'] = home_team_city
    game_pbp['aTeam_city'] = away_team_city
    game_pbp['game_id'] = game_id

    return game_pbp

def pbp_data(schedule):
    pbp_data_dict = {}
    game_ids = schedule['game_id'].tolist()
    schedule = schedule[schedule['completed'] == True]
    for game_id in game_ids:
        pbp_data = parse_game(schedule, game_id)
        if pbp_data is None:
            continue
        else:
            pbp_data_dict[game_id] = pbp_data
    return pbp_data_dict

def scrape_schedule():
    schedule_data = []
    url = 'https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json'
    r = requests.get(url)
    data = r.json()
    leagueSchedule = data['leagueSchedule']
    gameDates = leagueSchedule['gameDates']
    for game in gameDates:
        game_date = game['gameDate']
        games = game['games']
        for game in games:
            game_id = str(game['gameId'])
            game_status = game['gameStatus'] # 1 = upcoming, 2 = in progress, 3 = final
            game_status_text = game['gameStatusText']
            hTeam = game['homeTeam']
            hTeam_id = hTeam['teamId']
            hTeam_tricode = hTeam['teamTricode']
            hTeam_city = hTeam['teamCity']
            hTeam_name = hTeam['teamName']
            hTeam_score = hTeam['score']
            aTeam = game['awayTeam']
            aTeam_id = aTeam['teamId']
            aTeam_tricode = aTeam['teamTricode']
            aTeam_city = aTeam['teamCity']
            aTeam_name = aTeam['teamName']
            aTeam_score = aTeam['score']
            week_num = game['weekNumber']
            pre_season = week_num < 1
            completed = game_status == 3 and not pre_season
            in_progress = game_status == 2
            upcoming = game_status == 1
            schedule_data.append([game_id, game_date, game_status, game_status_text, hTeam_id, hTeam_tricode, hTeam_city, hTeam_name, hTeam_score, aTeam_id, aTeam_tricode, aTeam_city, aTeam_name, aTeam_score, week_num, pre_season, completed, in_progress, upcoming])
    schedule_data = pd.DataFrame(schedule_data, columns=['game_id', 'game_date', 'game_status', 'game_status_text', 'hTeam_id', 'hTeam_tricode', 'hTeam_city', 'hTeam_name', 'hTeam_score', 'aTeam_id', 'aTeam_tricode', 'aTeam_city', 'aTeam_name', 'aTeam_score', 'week_num', 'pre_season', 'completed', 'in_progress', 'upcoming'])
    schedule_data['game_id'] = schedule_data['game_id'].astype(str)
    schedule_data.set_index('game_id', inplace=True)
    # save to pickle
    return schedule_data