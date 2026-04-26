import pandas as pd
import csv
import os

os.system("clear")

# =========================================================
# FILE PATHS
# =========================================================

PLAYERS_PATH = "/Users/oktaycakim/Desktop/data_players.csv"
ORDER_PATH = "/Users/oktaycakim/Desktop/data_order.csv"
NEEDS_PATH = "/Users/oktaycakim/Desktop/data_needs.csv"

COLLEGE_PATHS = {
    "QB": "/Users/oktaycakim/Desktop/college_data/college_qb.csv",
    "WR": "/Users/oktaycakim/Desktop/college_data/college_wr.csv",
    "RB": "/Users/oktaycakim/Desktop/college_data/college_rb.csv",
    "TE": "/Users/oktaycakim/Desktop/college_data/college_te.csv",
    "DB": "/Users/oktaycakim/Desktop/college_data/college_db.csv",
    "S": "/Users/oktaycakim/Desktop/college_data/college_s.csv",
    "DL_LB": "/Users/oktaycakim/Desktop/college_data/college_lb+dl.csv"
}

# =========================================================
# POSITION CLEANING
# =========================================================

def normalize_position(pos):
    pos = str(pos).strip().upper()

    if pos in ["DE", "DT"]:
        return "DL"
    if pos in ["ILB", "OLB", "LB"]:
        return "LB"
    if pos in ["OT", "OG", "C"]:
        return "OL"
    if pos == "DB":
        return "CB"

    return pos

# =========================================================
# LOAD PLAYER / COMBINE DATA
# =========================================================

df = pd.read_csv(PLAYERS_PATH)
df.columns = df.columns.str.strip()

df["Name"] = df["Name"].astype(str).str.strip()
df["Position"] = df["Position"].astype(str).str.strip()

df = df.dropna(subset=["Name", "Position"])
df = df[df["Name"] != "Name"]

df["Position"] = df["Position"].apply(normalize_position)

numeric_cols = df.columns.difference(["Name", "Position", "Conference"])
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

name_to_position = dict(zip(df["Name"], df["Position"]))

# =========================================================
# PHYSICAL SCORE
# =========================================================

position_features = {
    "QB": ["Height", "Weight", "40 Yard", "Vertical Jump", "Broad Jump", "Throwing Velocity (mph)", "Std. Throwing Score"],
    "WR": ["Height", "Weight", "40 Yard", "Vertical Jump", "Broad Jump", "3 Cone", "Shuttle"],
    "RB": ["Height", "Weight", "40 Yard", "Vertical Jump", "Broad Jump", "Bench Press"],
    "TE": ["Height", "Weight", "40 Yard", "Vertical Jump", "Broad Jump", "Bench Press"],
    "CB": ["Height", "Weight", "40 Yard", "Vertical Jump", "Broad Jump", "3 Cone", "Shuttle"],
    "S": ["Height", "Weight", "40 Yard", "Vertical Jump", "Broad Jump", "Bench Press"],
    "LB": ["Height", "Weight", "40 Yard", "Vertical Jump", "Bench Press"],
    "DL": ["Height", "Weight", "Bench Press", "Vertical Jump", "Broad Jump"],
    "OL": ["Height", "Weight", "Bench Press", "Broad Jump", "Vertical Jump"]
}

lower_is_better = ["40 Yard", "3 Cone", "Shuttle"]

def normalize_series(series, invert=False):
    s = pd.to_numeric(series, errors="coerce")
    s = s.fillna(s.mean())

    if s.max() == s.min() or pd.isna(s.max()) or pd.isna(s.min()):
        return pd.Series([0.5] * len(s), index=s.index)

    s = (s - s.min()) / (s.max() - s.min())

    if invert:
        s = 1 - s

    return s

position_groups = {}

for pos, group in df.groupby("Position"):
    group = group.copy()
    features = position_features.get(pos, [])

    norm_cols = []

    for feature in features:
        if feature in group.columns:
            norm_col = feature + "_norm"
            group[norm_col] = normalize_series(
                group[feature],
                invert=feature in lower_is_better
            )
            norm_cols.append(norm_col)

    if norm_cols:
        group["physical_score"] = group[norm_cols].mean(axis=1)
    else:
        group["physical_score"] = 0.5

    position_groups[pos] = group

physical_lookup = {}

for group in position_groups.values():
    for _, row in group.iterrows():
        physical_lookup[row["Name"]] = row["physical_score"]

df["physical_score"] = df["Name"].map(physical_lookup).fillna(0.5)

# =========================================================
# LOAD COLLEGE DATA
# =========================================================

college_groups = {}

for key, path in COLLEGE_PATHS.items():
    temp = pd.read_csv(path)
    temp.columns = temp.columns.str.strip()
    temp["Name"] = temp["Name"].astype(str).str.strip()
    temp["Position"] = temp["Name"].map(name_to_position)
    college_groups[key] = temp

# =========================================================
# CONFERENCE SCORE
# =========================================================

def clean_conference(conf):
    if not isinstance(conf, str):
        return "UNKNOWN"

    conf = conf.strip().upper()

    mapping = {
        "BIG 10": "BIG TEN",
        "BIGTEN": "BIG TEN",
        "BIG 12": "BIG 12",
        "PAC 12": "PAC 12",
        "MVFC": "FCS",
        "CAA": "FCS",
        "OVC": "FCS",
        "MEAC": "FCS",
        "IVY": "FCS",
        "BIG SKY": "FCS",
        "STHLND": "FCS",
        "PTRT": "FCS",
        "NEC": "FCS",
        "MIAA": "FCS",
        "CIAA": "FCS",
        "GLIAC": "FCS",
        "GNAC": "FCS",
        "SOCON": "FCS"
    }

    return mapping.get(conf, conf)

conference_strength = {
    "SEC": 1.00,
    "BIG TEN": 0.95,
    "ACC": 0.90,
    "BIG 12": 0.90,
    "PAC 12": 0.90,
    "AAC": 0.80,
    "MOUNTAIN WEST": 0.75,
    "SUN BELT": 0.72,
    "C-USA": 0.70,
    "MAC": 0.70,
    "INDEPENDENT": 0.78,
    "FCS": 0.60,
    "UNKNOWN": 0.75
}

if "Conference" in df.columns:
    df["Conference_clean"] = df["Conference"].apply(clean_conference)
else:
    df["Conference_clean"] = "UNKNOWN"

name_to_conf = dict(zip(df["Name"], df["Conference_clean"]))

# =========================================================
# PRODUCTION SCORE
# =========================================================

def norm_any(data, columns):
    for col in columns:
        if col in data.columns:
            s = pd.to_numeric(data[col], errors="coerce").fillna(0)

            if s.max() == s.min():
                return pd.Series([0.5] * len(s), index=data.index)

            return (s - s.min()) / (s.max() - s.min())

    return pd.Series([0.5] * len(data), index=data.index)

def normalize_score(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0)

    if s.max() == s.min():
        return pd.Series([0.5] * len(s), index=s.index)

    return (s - s.min()) / (s.max() - s.min())

production_groups = {}

for pos in ["QB", "WR", "RB", "TE", "CB", "S", "LB", "DL"]:
    if pos == "CB":
        data = college_groups["DB"].copy()
        data = data[data["Position"] == "CB"]
    elif pos == "S":
        data = college_groups["S"].copy()
        data = data[data["Position"] == "S"]
    elif pos in ["LB", "DL"]:
        data = college_groups["DL_LB"].copy()
        data = data[data["Position"] == pos]
    else:
        data = college_groups[pos].copy()
        data = data[data["Position"] == pos]

    if len(data) == 0:
        continue

    if pos == "QB":
        prod = (
            0.25 * norm_any(data, ["Pass TD", "TD"]) +
            0.20 * norm_any(data, ["Pass Yds", "Yds"]) +
            0.25 * norm_any(data, ["Rush Yds", "Rushing Yards"]) +
            0.15 * norm_any(data, ["Y/A", "Yds/Att"]) +
            0.10 * norm_any(data, ["Completion %", "Comp %"]) -
            0.05 * norm_any(data, ["INT", "Interceptions"])
        )

    elif pos == "WR":
        prod = (
            0.35 * norm_any(data, ["Yds", "Rec Yds", "Receiving Yards"]) +
            0.30 * norm_any(data, ["TD", "Rec TD", "Receiving TD"]) +
            0.25 * norm_any(data, ["Y/R", "YPC", "Yards Per Reception"]) +
            0.10 * norm_any(data, ["Rec", "Receptions"])
        )

    elif pos == "RB":
        prod = (
            0.40 * norm_any(data, ["Rush Yds", "Rushing Yards"]) +
            0.35 * norm_any(data, ["Rush TD", "Rushing TD"]) +
            0.20 * norm_any(data, ["YPC", "Y/A"]) +
            0.05 * norm_any(data, ["Rec Yds", "Receiving Yards"])
        )

    elif pos == "TE":
        prod = (
            0.35 * norm_any(data, ["Yds", "Rec Yds", "Receiving Yards"]) +
            0.30 * norm_any(data, ["TD", "Rec TD", "Receiving TD"]) +
            0.20 * norm_any(data, ["Rec", "Receptions"]) +
            0.15 * norm_any(data, ["Y/R", "YPC"])
        )

    elif pos == "CB":
        prod = (
            0.35 * norm_any(data, ["PBUs", "PBU", "Pass Breakups"]) +
            0.30 * norm_any(data, ["INT", "Interceptions"]) +
            0.20 * norm_any(data, ["FF", "Forced Fumbles"]) +
            0.15 * norm_any(data, ["Tackles"])
        )

    elif pos == "S":
        prod = (
            0.40 * norm_any(data, ["INT", "Interceptions"]) +
            0.25 * norm_any(data, ["PBUs", "PBU", "Pass Breakups"]) +
            0.20 * norm_any(data, ["FF", "Forced Fumbles"]) +
            0.15 * norm_any(data, ["Tackles"])
        )

    elif pos == "LB":
        prod = (
            0.30 * norm_any(data, ["TFL"]) +
            0.25 * norm_any(data, ["Tackles"]) +
            0.20 * norm_any(data, ["FF", "Forced Fumbles"]) +
            0.15 * norm_any(data, ["Sacks"]) +
            0.10 * norm_any(data, ["INT", "Interceptions"])
        )

    elif pos == "DL":
        prod = (
            0.35 * norm_any(data, ["TFL"]) +
            0.30 * norm_any(data, ["Sacks"]) +
            0.20 * norm_any(data, ["FF", "Forced Fumbles"]) +
            0.15 * norm_any(data, ["Tackles"])
        )

    data["raw_production_score"] = prod
    data["production_score"] = normalize_score(data["raw_production_score"])

    data["conference_score"] = data["Name"].apply(
        lambda name: conference_strength.get(name_to_conf.get(name, "UNKNOWN"), 0.75)
    )

    data["production_plus_conf"] = (
        0.65 * data["production_score"] +
        0.35 * data["conference_score"]
    )

    production_groups[pos] = data

production_lookup = {}

for group in production_groups.values():
    for _, row in group.iterrows():
        production_lookup[row["Name"]] = row["production_plus_conf"]

df["production_score"] = df["Name"].map(production_lookup).fillna(0.5)

# =========================================================
# FINAL PLAYER SCORE
# =========================================================

def final_weighted_score(row):
    pos = row["Position"]
    physical = row["physical_score"]
    production = row["production_score"]
    synergy = physical * production

    if pos == "QB":
        return 0.50 * physical + 0.35 * production + 0.15 * synergy

    if pos in ["WR", "CB"]:
        return 0.50 * physical + 0.35 * production + 0.15 * synergy

    if pos == "S":
        return 0.40 * physical + 0.45 * production + 0.15 * synergy

    if pos in ["LB", "DL"]:
        return 0.35 * physical + 0.50 * production + 0.15 * synergy

    if pos == "RB":
        return 0.35 * physical + 0.55 * production + 0.10 * synergy

    if pos == "TE":
        return 0.40 * physical + 0.50 * production + 0.10 * synergy

    if pos == "OL":
        return 0.70 * physical + 0.30 * production

    return 0.45 * physical + 0.45 * production + 0.10 * synergy

df["final_score"] = df.apply(final_weighted_score, axis=1)

# =========================================================
# LOAD TEAM NEEDS
# =========================================================

def load_team_needs(path):
    team_needs = {}

    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            team = row["team"].strip()
            raw_needs = {}

            for pos, val in row.items():
                if pos == "team":
                    continue

                clean_pos = normalize_position(pos)
                raw_needs[clean_pos] = float(val)

            max_val = max(raw_needs.values())
            min_val = min(raw_needs.values())

            normalized_needs = {}

            for pos, val in raw_needs.items():
                if max_val == min_val:
                    normalized_needs[pos] = 0.5
                else:
                    normalized_needs[pos] = 0.2 + 0.8 * ((val - min_val) / (max_val - min_val))

            team_needs[team] = normalized_needs

    return team_needs

team_needs = load_team_needs(NEEDS_PATH)

# =========================================================
# LOAD DRAFT ORDER
# =========================================================

def load_draft_order(path):
    order_df = pd.read_csv(path)
    order_df.columns = order_df.columns.str.strip()

    order_df["Draft Order"] = pd.to_numeric(order_df["Draft Order"], errors="coerce")
    order_df["Team Drafting"] = order_df["Team Drafting"].astype(str).str.strip()

    order_df = order_df.dropna(subset=["Draft Order", "Team Drafting"])
    order_df["Draft Order"] = order_df["Draft Order"].astype(int)

    order_df = order_df.sort_values("Draft Order")

    draft_order = []

    for _, row in order_df.iterrows():
        draft_order.append({
            "pick": row["Draft Order"],
            "team": row["Team Drafting"]
        })

    return draft_order

draft_order = load_draft_order(ORDER_PATH)

# =========================================================
# DRAFT SIMULATION
# =========================================================

available_players = df[df["Position"].isin(["QB", "WR", "RB", "TE", "CB", "S", "LB", "DL", "OL"])].copy()

team_gains = {team: [] for team in team_needs}

def score_player_for_team(player_row, team):
    pos = player_row["Position"]
    player_score = player_row["final_score"]

    need_score = team_needs.get(team, {}).get(pos, 0.5)

    return player_score * need_score

def build_team_board(team, available):
    board = available.copy()

    board["team_board_score"] = board.apply(
        lambda row: score_player_for_team(row, team),
        axis=1
    )

    board = board.sort_values("team_board_score", ascending=False)

    return board

def make_pick(team, available):
    board = build_team_board(team, available)

    if len(board) == 0:
        return None

    return board.iloc[0]

draft_results = []

for pick_info in draft_order:
    pick_number = pick_info["pick"]
    team = pick_info["team"]

    if pick_number > 100:
        break

    selected = make_pick(team, available_players)

    if selected is None:
        break

    name = selected["Name"]
    pos = selected["Position"]

    draft_results.append({
        "Pick": pick_number,
        "Team": team,
        "Name": name,
        "Position": pos,
        "Final Score": selected["final_score"]
    })

    if team not in team_gains:
        team_gains[team] = []

    team_gains[team].append({
        "Name": name,
        "Position": pos,
        "Final Score": selected["final_score"]
    })

    if team in team_needs and pos in team_needs[team]:
        team_needs[team][pos] *= 0.25

    available_players = available_players[available_players["Name"] != name]

# =========================================================
# OUTPUT
# =========================================================

draft_df = pd.DataFrame(draft_results)

print("\n\n================ SIMULATED DRAFT ================")
print(draft_df.to_string(index=False))

print("\n\n================ TEAM DRAFT GAINS ================")

for team, players in team_gains.items():
    if not players:
        continue

    print(f"\n{team}:")
    for p in players:
        print(f"  - {p['Name']} ({p['Position']})")