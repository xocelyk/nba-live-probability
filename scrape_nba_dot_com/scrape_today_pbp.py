import pandas as pd
import requests
import json
from datetime import datetime
import pickle
from utils import request_pbp, parse_game, scrape_schedule

# probably delete this file

def date_to_dig_string(date):
    return str(date.strftime('%Y%m%d'))

schedule = scrape_schedule()
today_date = datetime.today().date()
schedule['date'] = pd.to_datetime(schedule['game_date']).dt.date
print(schedule.iloc[0]['date'])
print(today_date)
today_schedule = schedule[schedule['date'] == today_date]
yesterday_schedule = schedule[schedule['date'] == today_date - pd.Timedelta(days=1)]
yesterday_schedule = yesterday_schedule[yesterday_schedule['completed'] == True]
print(yesterday_schedule)


