import random

import numpy as np
import pandas as pd
from team_info import team_result
# from eko_score import eko_score
from match import get_matches
from bs4 import BeautifulSoup
import urllib3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

RANGE = 2012, 2021

# Get all teams participating in Premier League:
league_url = "https://www.11v11.com/premier-league/"
http = urllib3.PoolManager()
r = http.request('GET', league_url)
assert r.status == 200
soup = BeautifulSoup(r.data, features="lxml")
links = soup.find_all("ul", {"class": "team-links"})[0]

# Get all participants match history:
history = dict()
for match in links:
    teamname = match.string.strip("\n")
    if teamname:
        history[teamname] = team_result(teamname, range(*RANGE))

# EKO Scores:
# eko_scores = {
# team: eko_score(frame, pd.Timestamp(year=RANGE[1], month=1, day=1), det=-1.005)
# for team, frame in history.items()
# }
# [print(f"{team}: {score}") for team, score in eko_scores.items()]

# Get all participants match tails and anonymyze them:
all_tails = list()
for team, hist in history.items():
    df = get_matches(team_history=hist, owner=team)
    all_tails.append(df)

all_matches = pd.concat(all_tails)

# Split 3:1
train_idx = random.sample(range(0, len(all_matches)-1), len(all_matches)*3//4)
test_idx = [i for i in range(0, len(all_matches)-1) if i not in train_idx]


train_X, train_y, test_X, test_y = (
    all_matches.iloc[train_idx, -1].to_list(),
    all_matches.iloc[train_idx, 0].to_list(),
    all_matches.iloc[test_idx, -1].to_list(),
    all_matches.iloc[test_idx, 0].to_list(),
)
for s in (train_X, train_y, test_X, test_y):
    print(np.shape(s))

# Get a model to train:
try:
    model = load_model("keras_model_v1.0")
except FileNotFoundError:
    model = Sequential([
        Dense(10, activation='relu'),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_X, train_y, epochs=10)
    model.save("keras_model_v1.0")

model.evaluate(test_X, test_y)
