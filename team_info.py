# Todo: detach getting the data from site into separata async function
import urllib3
import pandas as pd
import os
from typing import Iterable


def team_result(team: str, years: Iterable[int]):
    years_df = pd.DataFrame({
        "Date": pd.Series(dtype="datetime64[ns]"),
        "Match": pd.Series(dtype="object"),
        "Result": pd.Series(dtype=pd.CategoricalDtype(["W", "D", "L"], ordered=False))
    })

    for year in years:
        # Does directory exist?
        if not os.path.isdir(team):
            os.mkdir(team)

        # Test if HTML file is downloaded:
        try:
            open(os.path.join(team, str(year)) + '.html', 'r')

        except FileNotFoundError:
            # Download and save file:
            http = urllib3.PoolManager()
            convertname = team.lower().replace(" ", "-")
            r = http.request('GET', f"https://www.11v11.com/teams/{convertname}/tab/matches/season/{str(year)}/")
            assert r.status == 200

            with open(os.path.join(team, str(year)) + '.html', 'w') as site:
                site.write(r.data.decode('utf-8'))

        df = pd.read_html(os.path.join(team, str(year)) + '.html')[0]

        # Clean data
        def match_to_competitors(matchstr, host):
            """Return the team that was NOT the 'host'"""
            teams = matchstr.split(' v ')
            return teams[0] if teams[1] == host else teams[1]

        df["Match"] = df["Match"].apply(match_to_competitors, host=team)
        df["Date"] = pd.to_datetime(df["Date"])

        df = (df.dropna()
              .drop(columns=["Score", "Competition"])
              )

        # Append to dataframe:
        years_df = years_df.append(df)

    # Index by date and return
    return years_df.set_index("Date")
