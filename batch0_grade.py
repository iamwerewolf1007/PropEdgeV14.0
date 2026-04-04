"""
PropEdge V14.0 — batch0_grade.py
Batch 0: Grade yesterday → append gamelogs → H2H rebuild → DVP update → retrain → git push.
Runs at 08:00 UK. All steps are sequential and order-critical.
"""

import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    VERSION, FILE_GL_2526, FILE_H2H, FILE_TODAY, FILE_SEASON_2526,
    FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST, FILE_DVP,
    TEAM_ABBR, get_uk, today_et, clean_json, GIT_REMOTE, REPO_DIR,
)
from rolling_engine import filter_played, compute_rolling_for_new_rows
from h2h_builder import build_h2h
from dvp_updater import compute_and_save_dvp
from model_trainer import train_and_save
from reasoning_engine import generate_post_match_reason
from audit import log_event, verify_no_deletion
from batch_predict import git_push

# City/full-name → abbreviation map for NBA API opponent field
_CITY_TO_ABBR = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BKN", "Charlotte": "CHA",
    "Chicago": "CHI", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
    "Detroit": "DET", "Golden State": "GSW", "Houston": "HOU", "Indiana": "IND",
    "LA": "LAC", "Los Angeles": "LAL", "Memphis": "MEM", "Miami": "MIA",
    "Milwaukee": "MIL", "Minnesota": "MIN", "New Orleans": "NOP", "New York": "NYK",
    "Oklahoma City": "OKC", "Orlando": "ORL", "Philadelphia": "PHI", "Phoenix": "PHX",
    "Portland": "POR", "Sacramento": "SAC", "San Antonio": "SAS", "Toronto": "TOR",
    "Utah": "UTA", "Washington": "WAS",
}
# Also merge TEAM_ABBR (full name → abbr)
for full, abbr in TEAM_ABBR.items():
    _CITY_TO_ABBR[full] = abbr


def _city_to_abbr(city: str) -> str:
    """Convert NBA API city/teamCity to abbreviation."""
    if not city:
        return ""
    # Direct lookup
    if city in _CITY_TO_ABBR:
        return _CITY_TO_ABBR[city]
    # Try prefix match (e.g. "Los Angeles Lakers" → "Los Angeles" → "LAL")
    for k, v in _CITY_TO_ABBR.items():
        if city.startswith(k):
            return v
    return city  # return as-is if no match


# ─────────────────────────────────────────────────────────────────────────────
# NBA API HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_min(v) -> float:
    s = str(v).strip()
    if s in ("", "None", "nan", "0", "PT00M00.00S"):
        return 0.0
    if s.startswith("PT") and "M" in s:
        m = re.match(r"PT(\d+)M([\d.]+)S", s)
        return float(m.group(1)) + float(m.group(2)) / 60 if m else 0.0
    if ":" in s:
        p = s.split(":")
        return float(p[0]) + float(p[1]) / 60
    try:
        return float(s)
    except Exception:
        return 0.0


# Browser-like headers required by stats.nba.com
_NBA_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
}


def _fetch_from_csv(date_str: str) -> tuple[list[dict], set[str]] | tuple[None, None]:
    """
    Fallback: read box scores for date_str directly from the game log CSV.
    Used when the NBA API is unavailable (timeout, rate-limit, etc.).
    Returns (played_rows, players_in_box) if data found, else (None, None).
    """
    try:
        gl = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
        day = gl[gl["GAME_DATE"].dt.strftime("%Y-%m-%d") == date_str]
        played = day[(day["DNP"].fillna(0) == 0) & (day["MIN_NUM"].fillna(0) > 0)]
        dnp_stubs = day[(day["DNP"].fillna(0) == 1)]
        if len(played) == 0:
            latest = gl["GAME_DATE"].max().date() if len(gl) else "unknown"
            print(f"  ℹ  CSV fallback: no played rows for {date_str} (CSV latest: {latest})")
            return None, None
        played_rows: list[dict] = []
        players_in_box: set[str] = set()
        # Include DNP stub names in players_in_box so they are NOT falsely tagged as DNP
        # (they are in the CSV as confirmed DNP, not absent from the box entirely)
        for _, r in dnp_stubs.iterrows():
            pname = str(r.get("PLAYER_NAME", "")).strip()
            if pname:
                players_in_box.add(pname)
        for _, r in played.iterrows():
            pname = str(r.get("PLAYER_NAME", "")).strip()
            if not pname:
                continue
            players_in_box.add(pname)
            played_rows.append({
                "PLAYER_NAME":  pname,
                "GAME_DATE":    date_str,
                "PTS":          float(r.get("PTS", 0) or 0),
                "MIN_NUM":      float(r.get("MIN_NUM", 0) or 0),
                "FGA":          float(r.get("FGA", 0) or 0),
                "FGM":          float(r.get("FGM", 0) or 0),
                "FG3A":         float(r.get("FG3A", 0) or 0),
                "FG3M":         float(r.get("FG3M", 0) or 0),
                "FTA":          float(r.get("FTA", 0) or 0),
                "FTM":          float(r.get("FTM", 0) or 0),
                "REB":          float(r.get("REB", 0) or 0),
                "AST":          float(r.get("AST", 0) or 0),
                "STL":          float(r.get("STL", 0) or 0),
                "BLK":          float(r.get("BLK", 0) or 0),
                "TOV":          float(r.get("TOV", 0) or 0),
                "PLUS_MINUS":   float(r.get("PLUS_MINUS", 0) or 0),
                "DNP":          0,
                "OPPONENT":     str(r.get("OPPONENT", "") or ""),
                "IS_HOME":      int(r.get("IS_HOME", 0) or 0),
            })
        print(f"  CSV fallback: {len(played_rows)} played rows, {len(players_in_box)} players")
        log_event("B0", "CSV_FALLBACK_USED", detail=f"{len(played_rows)} rows for {date_str}")
        return played_rows, players_in_box
    except Exception as e:
        print(f"  ⚠ CSV fallback failed: {e}")
        return None, None


def fetch_boxscores(date_str: str) -> tuple[list[dict], set[str]]:
    """
    Fetch yesterday's box scores via nba_api, with automatic CSV fallback.

    Priority:
      1. NBA API (nba_api ScoreboardV3 + BoxScoreTraditionalV3)
         — 3 retries with increasing backoff (5s, 10s, 20s)
      2. Game log CSV fallback — used automatically if the API fails
         (timeout, rate-limit, network block from UK IP, etc.)
         Requires that batch0_grade already appended the previous day.
         NOTE: CSV fallback skips append_gamelogs (data already in CSV).

    Returns (played_rows, players_in_box).
    Returns (None, None) only if BOTH the API and the CSV have no data.
    """
    from nba_api.stats.endpoints import scoreboardv3, boxscoretraditionalv3
    import time

    print(f"  Fetching box scores for {date_str}...")
    played_rows: list[dict] = []
    players_in_box: set[str] = set()

    # --- Step 1: NBA API with retries and longer backoff ---
    games = []
    last_err = None
    for attempt in range(3):
        try:
            if attempt > 0:
                wait = 5 * (2 ** (attempt - 1))   # 5s, 10s
                print(f"  Retry {attempt}/2 (waiting {wait}s)...")
                time.sleep(wait)
            sb = scoreboardv3.ScoreboardV3(
                game_date=date_str,
                league_id="00",
                headers=_NBA_HEADERS,
                timeout=60,
            ).get_normalized_dict()
            games = sb.get("scoreboard", {}).get("games", [])
            last_err = None
            break
        except Exception as e:
            last_err = e
            print(f"  ⚠ ScoreboardV3 attempt {attempt+1} error: {e}")

    # --- Step 2: If API failed, try CSV fallback ---
    if last_err is not None:
        log_event("B0", "FETCH_FAILED", detail=f"ScoreboardV3 error after retries: {last_err}")
        print(f"  ⚠ NBA API failed — trying CSV fallback...")
        csv_rows, csv_players = _fetch_from_csv(date_str)
        if csv_rows is not None:
            return csv_rows, csv_players
        print(f"  ⚠ CSV fallback also failed — aborting grade to prevent false DNPs.")
        print(f"  ℹ  When CSV is updated, run: python3 run.py grade-csv --date {date_str}")
        return None, None

    if not games:
        log_event("B0", "FETCH_EMPTY_WARNING", detail=f"0 games on scoreboard for {date_str}")
        print(f"  ⚠ 0 games found for {date_str} — NBA off-day or API issue")
        print(f"  ℹ  If this was NOT an off-day, try: python3 run.py grade --date {date_str}")
        return None, None

    # --- Step 2: Get each box score ---
    for game in games:
        gid = game.get("gameId") or game.get("id", "")
        if not gid:
            continue
        try:
            time.sleep(0.6)
            bx = boxscoretraditionalv3.BoxScoreTraditionalV3(
                game_id=gid,
                headers=_NBA_HEADERS,
                timeout=45,
            ).get_normalized_dict()

            for team_key in ("homeTeam", "awayTeam"):
                team_data = bx.get(team_key, {})
                opp_city = team_data.get("teamCity", "")
                opp_abbr = _city_to_abbr(opp_city)

                for player in team_data.get("players", []):
                    pname = str(player.get("name", "")).strip()
                    if not pname:
                        continue
                    players_in_box.add(pname)
                    mins = _parse_min(
                        player.get("statistics", {}).get("minutesCalculated", 0)
                    )
                    if mins <= 0:
                        continue  # DNP-CD
                    s = player.get("statistics", {})
                    played_rows.append({
                        "PLAYER_NAME":  pname,
                        "GAME_DATE":    date_str,
                        "PTS":          float(s.get("points", 0) or 0),
                        "MIN_NUM":      round(mins, 2),
                        "FGA":          float(s.get("fieldGoalsAttempted", 0) or 0),
                        "FGM":          float(s.get("fieldGoalsMade", 0) or 0),
                        "FG3A":         float(s.get("threePointersAttempted", 0) or 0),
                        "FG3M":         float(s.get("threePointersMade", 0) or 0),
                        "FTA":          float(s.get("freeThrowsAttempted", 0) or 0),
                        "FTM":          float(s.get("freeThrowsMade", 0) or 0),
                        "REB":          float(s.get("reboundsTotal", 0) or 0),
                        "AST":          float(s.get("assists", 0) or 0),
                        "STL":          float(s.get("steals", 0) or 0),
                        "BLK":          float(s.get("blocks", 0) or 0),
                        "TOV":          float(s.get("turnovers", 0) or 0),
                        "PLUS_MINUS":   float(s.get("plusMinusPoints", 0) or 0),
                        "DNP":          0,
                        "OPPONENT":     opp_abbr,   # ← abbreviation, not city name
                        "IS_HOME":      1 if team_key == "homeTeam" else 0,
                    })
        except Exception as e:
            print(f"  ⚠ BoxScore error game {gid}: {e}")

    log_event("B0", "BOXSCORES_FETCHED",
              detail=f"{len(played_rows)} rows, {len(players_in_box)} players")
    print(f"  Box scores: {len(played_rows)} played rows, "
          f"{len(players_in_box)} players in box")
    return played_rows, players_in_box


# ─────────────────────────────────────────────────────────────────────────────
# GRADE PLAYS
# ─────────────────────────────────────────────────────────────────────────────

def grade_plays(
    date_str: str,
    played_rows: list[dict],
    players_in_box: set[str],
) -> tuple[list[str], list[dict]]:
    """
    Grade plays for date_str.
    GUARD: if both played_rows and players_in_box are empty AND we got here
    via a legitimate empty-day (not API failure), skip grading entirely.
    API failures return None/None upstream and this function is never called.
    """
    results_map = {r["PLAYER_NAME"]: r for r in played_rows}

    def _load(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path) as f:
            return json.load(f)

    graded_count = wins = losses = dnps = 0
    dnp_names: list[str] = []
    plays_for_check: list[dict] = []

    for path in (FILE_TODAY, FILE_SEASON_2526):
        data = _load(path)
        changed = False
        for play in data:
            if play.get("date") != date_str:
                continue
            if play.get("result") in ("WIN", "LOSS", "DNP"):
                continue  # immutable

            pname = play.get("player", "")
            line  = float(play.get("line", 20))
            dr    = str(play.get("direction", ""))
            is_over  = "OVER"  in dr.upper() and "LEAN" not in dr.upper()
            is_under = "UNDER" in dr.upper() and "LEAN" not in dr.upper()

            box = results_map.get(pname)

            if pname not in players_in_box or box is None:
                # Genuine DNP (not in box at all)
                play["result"]     = "DNP"
                play["actualPts"]  = None
                play["actualMin"]  = 0
                post, lt = generate_post_match_reason(play)
                play["postMatchReason"] = post
                play["lossType"]   = "DNP"
                dnps += 1
                dnp_names.append(pname)
            else:
                actual     = float(box.get("PTS", 0))
                actual_min = float(box.get("MIN_NUM", 0))
                play["actualPts"] = actual
                play["actualMin"] = round(actual_min, 1)
                play["delta"]     = round(actual - line, 1)

                if is_over:
                    win = actual > line
                elif is_under:
                    win = actual <= line
                else:
                    win = (actual > line and "OVER" in dr) or \
                          (actual <= line and "UNDER" in dr)

                play["result"] = "WIN" if win else "LOSS"
                if win: wins += 1
                else:   losses += 1

                box_data = {
                    "actual_pts": actual,
                    "actual_min": actual_min,
                    "actual_fga": box.get("FGA", 0),
                    "actual_fgm": box.get("FGM", 0),
                    "actual_fg_pct": box.get("FGM", 0) / max(box.get("FGA", 1), 1),
                }
                post, lt = generate_post_match_reason(play, box_data)
                play["postMatchReason"] = post
                play["lossType"] = lt
                plays_for_check.append(play)

            graded_count += 1
            changed = True

        if changed:
            with open(path, "w") as f:
                json.dump(clean_json(data), f, indent=2)

    log_event("B0", "BATCH_SUMMARY",
              detail=f"graded={graded_count} wins={wins} losses={losses} dnp={dnps}")
    print(f"  Graded: {graded_count} | W:{wins} L:{losses} DNP:{dnps}")
    return dnp_names, plays_for_check


# ─────────────────────────────────────────────────────────────────────────────
# APPEND GAME LOGS
# ─────────────────────────────────────────────────────────────────────────────

def append_gamelogs(
    played_rows: list[dict], dnp_names: list[str], date_str: str
) -> None:
    existing = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
    rows_before = len(existing)
    log_event("B0", "FILE_STATE_BEFORE_APPEND", str(FILE_GL_2526.name),
              rows_before, rows_before)

    bio_cache: dict[str, dict] = {}
    for _, r in existing.drop_duplicates("PLAYER_NAME", keep="last").iterrows():
        bio_cache[r["PLAYER_NAME"]] = r.to_dict()

    new_rows = []
    for r in played_rows:
        row = dict(r)
        row["GAME_DATE"] = date_str
        row["DNP"] = 0
        new_rows.append(row)

    for pname in set(dnp_names):
        bio = bio_cache.get(pname, {})
        stub = {
            "PLAYER_NAME": pname, "GAME_DATE": date_str,
            "PTS": float("nan"), "MIN_NUM": 0,
            "FGA": float("nan"), "FGM": float("nan"),
            "FG3A": float("nan"), "FG3M": float("nan"),
            "FTA": float("nan"), "FTM": float("nan"),
            "DNP": 1,
            "PLAYER_POSITION":        bio.get("PLAYER_POSITION", ""),
            "GAME_TEAM_ABBREVIATION": bio.get("GAME_TEAM_ABBREVIATION", ""),
            "USAGE_APPROX":           float("nan"),
            "IS_HOME":                bio.get("IS_HOME", 0),
            "OPPONENT":               bio.get("OPPONENT", ""),
        }
        new_rows.append(stub)

    new_df = pd.DataFrame(new_rows)
    new_df["GAME_DATE"] = pd.to_datetime(new_df["GAME_DATE"])

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined  = compute_rolling_for_new_rows(combined)
    combined  = combined.drop_duplicates(
        subset=["PLAYER_NAME", "GAME_DATE"], keep="last"
    )
    combined  = combined.sort_values(
        ["PLAYER_NAME", "GAME_DATE"]
    ).reset_index(drop=True)
    combined.to_csv(FILE_GL_2526, index=False)

    verify_no_deletion(FILE_GL_2526, rows_before, "B0")
    print(f"  Gamelogs: {rows_before} → {len(combined)} rows "
          f"(+{len(combined)-rows_before})")


# ─────────────────────────────────────────────────────────────────────────────
# POST-MATCH ROLLING UPDATE
# ─────────────────────────────────────────────────────────────────────────────

def update_postmatch_rolling(date_str: str) -> None:
    from rolling_engine import filter_played
    gl = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
    played = filter_played(gl)
    cutoff = pd.Timestamp(date_str) + timedelta(days=1)

    player_idx = {
        pname: grp.sort_values("GAME_DATE").reset_index(drop=True)
        for pname, grp in played.groupby("PLAYER_NAME")
    }

    for path in (FILE_TODAY, FILE_SEASON_2526):
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        changed = False
        for play in data:
            if play.get("date") != date_str:
                continue
            if play.get("result") == "DNP":
                continue
            pname = play.get("player", "")
            hist  = player_idx.get(pname)
            if hist is None:
                continue
            prior = hist[hist["GAME_DATE"] < cutoff]
            if len(prior) < 2:
                continue
            pts = prior["PTS"].values.astype(float)
            play["l30"] = round(float(np.mean(pts[-30:])), 1)
            play["l10"] = round(float(np.mean(pts[-10:])), 1)
            play["l5"]  = round(float(np.mean(pts[-5:])),  1)
            play["l3"]  = round(float(np.mean(pts[-3:])),  1)
            changed = True
        if changed:
            with open(path, "w") as f:
                json.dump(clean_json(data), f, indent=2)
    print("  Post-match rolling updated.")


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING CROSSCHECK
# ─────────────────────────────────────────────────────────────────────────────

def crosscheck_rolling_stats(
    plays: list[dict], date_str: str
) -> dict[str, str | None]:
    from rolling_engine import filter_played
    gl = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
    played = filter_played(gl)
    player_idx = {
        pname: grp.sort_values("GAME_DATE").reset_index(drop=True)
        for pname, grp in played.groupby("PLAYER_NAME")
    }
    integrity: dict[str, str | None] = {}

    for play in plays:
        pname = play.get("player", "")
        stored = float(play.get("l30", 0) or 0)
        hist   = player_idx.get(pname)
        if hist is None:
            continue
        prior = hist[hist["GAME_DATE"] < pd.Timestamp(date_str)]
        if len(prior) < 5:
            continue
        fresh = float(np.mean(prior["PTS"].values[-30:]))
        dev   = abs(fresh - stored)
        if dev > 1.0:
            msg = (f"L30 drift {dev:.2f}pts "
                   f"(stored {stored:.1f}, fresh {fresh:.1f})")
            log_event("B0", "ROLLING_CROSSCHECK_FAIL", pname, detail=msg)
            integrity[pname] = msg
        else:
            integrity[pname] = None

    fails = sum(1 for v in integrity.values() if v)
    print(f"  Rolling crosscheck: {fails} flag(s) of {len(integrity)} checked")
    return integrity


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BATCH 0
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PropEdge V14 Batch 0 — Grade + Retrain")
    parser.add_argument("--date", type=str, default=None,
                        help="Grade a specific date (YYYY-MM-DD). Default: yesterday.")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Skip model retraining after grading.")
    # Parse only known args so run.py can pass extra args safely
    args, _ = parser.parse_known_args()

    if args.date:
        yesterday = args.date
        print(f"  ⚠ Manual date override: grading {yesterday}")
    else:
        yesterday_ts = datetime.now(get_uk()) - timedelta(days=1)
        yesterday    = yesterday_ts.strftime("%Y-%m-%d")

    no_retrain = getattr(args, 'no_retrain', False)

    print(f"\n{'='*60}")
    print(f"  PropEdge {VERSION} — Batch 0 (Grade + Retrain)")
    print(f"  Grading: {yesterday}")
    print(f"{'='*60}")
    log_event("B0", "BATCH_START", detail=f"grading_date={yesterday}")

    # 1. Fetch box scores — NBA API with automatic CSV fallback
    played_rows, players_in_box = fetch_boxscores(yesterday)

    if played_rows is None:
        # Both NBA API and CSV fallback failed — nothing to grade
        log_event("B0", "GRADING_ABORTED",
                  detail="API + CSV fallback both failed — grading skipped")
        print("  ⚠ GRADING ABORTED: NBA API timed out and CSV has no data yet.")
        print(f"  ℹ  Once the game log CSV is updated, run:")
        print(f"       python3 run.py grade-csv --date {yesterday}")
        return

    # 2. Grade plays
    dnp_names, plays_for_check = grade_plays(
        yesterday, played_rows, players_in_box
    )

    # 3. Append game logs — skip if date already in CSV (CSV fallback case)
    existing_gl = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
    already_in_csv = (existing_gl["GAME_DATE"].dt.strftime("%Y-%m-%d") == yesterday).any()
    if already_in_csv:
        print(f"  Game log already has rows for {yesterday} — skipping append (CSV fallback used)")
    elif played_rows or dnp_names:
        append_gamelogs(played_rows, dnp_names, yesterday)

    # 3b. Post-game rolling refresh
    update_postmatch_rolling(yesterday)

    # 4. Rolling crosscheck
    integrity = crosscheck_rolling_stats(plays_for_check, yesterday)

    # 5. Apply integrity flags
    for path in (FILE_TODAY, FILE_SEASON_2526):
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        changed = False
        for play in data:
            if play.get("date") != yesterday:
                continue
            flag = integrity.get(play.get("player", ""))
            if flag:
                play["postMatchReason"] = (
                    play.get("postMatchReason", "") +
                    f" ⚠ Data integrity: {flag}"
                )
                changed = True
        if changed:
            with open(path, "w") as f:
                json.dump(clean_json(data), f, indent=2)

    # 6. Rebuild H2H
    from config import FILE_GL_2425
    build_h2h(FILE_GL_2425, FILE_GL_2526, FILE_H2H)

    # 7. Update DVP
    compute_and_save_dvp(FILE_GL_2526, FILE_DVP)

    # 8. Retrain (skip if --no-retrain)
    if no_retrain:
        print("  --no-retrain: skipping model retraining.")
    else:
        print("  Retraining V14 models (~5-8 min)...")
        train_and_save(FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST)

    # 9. Git push
    git_push(f"B0: grade {yesterday}")

    log_event("B0", "BATCH_COMPLETE", detail=f"grade_date={yesterday}")
    print(f"\n  ✓ Batch 0 complete — graded {yesterday}.\n")


if __name__ == "__main__":
    main()
