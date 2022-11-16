# scrape the nba schedule json file

import requests
import json
import os
import pandas as pd
import csv
import pickle

### moving this all to utils

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
    # save to pickle
    return schedule_data

if __name__ == '__main__':
    schedule = scrape_schedule()
    with open('../nba_dot_com_data/schedule_2023.pickle', 'wb') as f:
        pickle.dump(schedule, f)


