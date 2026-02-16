import json
import pandas as pd
from json_return import json_files, json_id

all_matches = []

def innings_runs(match_data: dict, innings_index: int) -> int:
    """Return total runs for an innings (sum of ball-level total runs)."""
    innings = match_data.get("innings", [])
    if innings_index >= len(innings):
        return 0

    overs = innings[innings_index].get("overs", [])
    total = 0
    for over in overs:
        for ball in over.get("deliveries", []):
            total += int(ball.get("runs", {}).get("total", 0))
    return total

def innings_wickets(match_data: dict, innings_index: int) -> int:
    """Return wickets lost for an innings (count wicket events in deliveries)."""
    innings = match_data.get("innings", [])
    if innings_index >= len(innings):
        return 0

    overs = innings[innings_index].get("overs", [])
    wkts = 0
    for over in overs:
        for ball in over.get("deliveries", []):
            wkts += len(ball.get("wickets", []))
    return wkts


# Build a match-level dataframe: one row per JSON file
for json_path, match_id in zip(json_files, json_id):
    # Load match JSON
    with open(json_path, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    # Extract match info 
    info = match_data.get("info", {})
    teams = info.get("teams", [None, None])
    city = info.get("city")
    venue = info.get("venue")
    date = info.get("dates")

    # Outcome can be winner or result (tie/no result/etc.)
    outcome = info.get("outcome", {})
    winner = outcome.get("winner") or outcome.get("result")

    # Compute innings totals
    runs_1 = innings_runs(match_data, 0)
    runs_2 = innings_runs(match_data, 1)
    wkts_1 = innings_wickets(match_data, 0)
    wkts_2 = innings_wickets(match_data, 1)

    all_matches.append({
        "ID": match_id,
        "City": city,
        "Venue": venue,
        "Team A": teams[0] if len(teams) > 0 else None,
        "First Inning": runs_1,
        "First Inn Wickets": wkts_1,
        "Team B": teams[1] if len(teams) > 1 else None,
        "Second Inning": runs_2,
        "Second Inn Wickets": wkts_2,
        "Winner": winner,
    })

# Create dataframe
df = pd.DataFrame(all_matches)

def show_dataframe(n: int = 5):
    """Print a small preview of the dataframe."""
    print(df.head(n))
    print("\nshape:", df.shape)

