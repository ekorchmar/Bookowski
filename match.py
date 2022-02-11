from collections import namedtuple
import pandas as pd

match_metadata = namedtuple("MatchMD", ["team1", "team2", "date"])
TAIL = 10
scoring = {
    "W": 1,
    "D": 0,
    "L": -1
}


# Match is an ordinal series object of last N wins, draws and losses in temporal order,
# labeled as -1, 0, or 1 depending on its own result.


# Splice team history into matches:
def get_matches(team_history: pd.DataFrame, owner: str, tail=TAIL) -> pd.DataFrame:
    portraits = []
    for index, entry in enumerate(team_history.iterrows()):

        if index < tail+1:
            continue

        # Get metadata:
        team1, team2, result = owner, entry[1]["Match"], scoring[entry[1]["Result"]]
        metadata = match_metadata(team1, team2, entry[0])
        # dublicate = match_metadata(team2, team1, entry[0])

        # Test:
        if result == 0:
            continue

        # Get payload:
        previousN = team_history.iloc[index - tail-1:index-1, 1].apply(scoring.get).to_list()

        # Transform to one-hot vector for TensorFlow
        def one_hot(val: int, inputs=(-1, 0, 1)):
            out = [0] * len(inputs)
            idx = inputs.index(val)
            out[idx] = 1
            return out

        # Form an entry:
        match_portrait: dict = {
            "label": one_hot(result),
            "metadata": metadata,
            # "dublicate_md": dublicate,
            "tail": previousN
        }
        portraits.append(match_portrait)

        if len(previousN) < 10:
            print(f"indices: {index - tail - 1, index - 1}")

    return pd.DataFrame(portraits)
