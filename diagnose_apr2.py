"""
PropEdge V14.0 — diagnose_apr2.py
Run this on your machine to understand why Apr 2 data is missing.

Usage: python3 diagnose_apr2.py
       python3 diagnose_apr2.py --date 2026-04-02
"""

import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--date", default="2026-04-02", help="Date to diagnose (YYYY-MM-DD)")
args, _ = parser.parse_known_args()
TARGET = args.date

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import FILE_TODAY, FILE_SEASON_2526, FILE_GL_2526, FILE_PROPS

import pandas as pd

SEP = "=" * 65

print(SEP)
print(f"  PropEdge V14 — Root Cause Diagnostic for {TARGET}")
print(SEP)

# ── 1. today.json ──────────────────────────────────────────────
print(f"\n[1] today.json — {FILE_TODAY}")
if not FILE_TODAY.exists():
    print("  ✗ FILE MISSING")
else:
    with open(FILE_TODAY) as f:
        today = json.load(f)
    all_dates  = sorted(set(p["date"] for p in today))
    target_t   = [p for p in today if p["date"] == TARGET]
    print(f"  Total plays   : {len(today)}")
    print(f"  Dates present : {all_dates}")
    print(f"  {TARGET} plays : {len(target_t)}")
    if target_t:
        results = {}
        for p in target_t:
            r = p.get("result") or "ungraded"
            results[r] = results.get(r, 0) + 1
        tiers = {}
        for p in target_t:
            t = p.get("tierLabel", "?")
            tiers[t] = tiers.get(t, 0) + 1
        print(f"  Results  : {results}")
        print(f"  Tiers    : {tiers}")

# ── 2. season_2025_26.json ─────────────────────────────────────
print(f"\n[2] season_2025_26.json — {FILE_SEASON_2526}")
if not FILE_SEASON_2526.exists():
    print("  ✗ FILE MISSING")
    season_target = []
else:
    with open(FILE_SEASON_2526) as f:
        season = json.load(f)
    s_dates      = sorted(set(p["date"] for p in season))
    season_target= [p for p in season if p["date"] == TARGET]
    print(f"  Total plays   : {len(season):,}")
    print(f"  Latest dates  : {s_dates[-5:]}")
    print(f"  {TARGET} plays : {len(season_target)}")
    if season_target:
        results = {}
        tiers   = {}
        for p in season_target:
            r = p.get("result") or "ungraded"
            t = p.get("tierLabel", "?")
            results[r] = results.get(r, 0) + 1
            tiers[t]   = tiers.get(t, 0) + 1
        print(f"  Results  : {results}")
        print(f"  Tiers    : {tiers}")
        print(f"  Sample plays:")
        for p in season_target[:4]:
            print(f"    {p['player']:30s} | {p.get('direction','?'):12s} {p.get('line','?'):4} "
                  f"| {p.get('tierLabel','?'):10s} | result={p.get('result','—'):8s} | pts={p.get('actualPts','—')}")

# ── 3. Game log CSV ────────────────────────────────────────────
print(f"\n[3] nba_gamelogs_2025_26.csv — {FILE_GL_2526}")
if not FILE_GL_2526.exists():
    print("  ✗ FILE MISSING")
    gl_played = []
else:
    gl = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
    target_gl   = gl[gl["GAME_DATE"].dt.strftime("%Y-%m-%d") == TARGET]
    gl_played   = target_gl[(target_gl["DNP"].fillna(0) == 0) & (target_gl["MIN_NUM"].fillna(0) > 0)]
    gl_dnp      = target_gl[target_gl["DNP"].fillna(0) == 1]
    print(f"  Total rows    : {len(gl):,}")
    print(f"  Latest date   : {gl['GAME_DATE'].max().date()}")
    print(f"  {TARGET} total   : {len(target_gl)}")
    print(f"  {TARGET} played  : {len(gl_played)}")
    print(f"  {TARGET} DNP     : {len(gl_dnp)}")
    if len(gl_played) > 0:
        games = gl_played["OPPONENT"].unique()
        print(f"  Opponents in log: {sorted(games)}")
        print(f"  Sample rows:")
        for _, r in gl_played.head(4).iterrows():
            print(f"    {r['PLAYER_NAME']:28s} | {r['OPPONENT']:4s} | PTS={r['PTS']:4.0f} | MIN={r['MIN_NUM']:4.0f}")

# ── 4. Excel props ─────────────────────────────────────────────
print(f"\n[4] Excel props — {FILE_PROPS}")
xl_target = pd.DataFrame()
if not FILE_PROPS.exists():
    print("  ✗ FILE MISSING")
else:
    xl = pd.read_excel(FILE_PROPS, sheet_name="Player_Points_Props")
    xl["Date"] = pd.to_datetime(xl["Date"])
    xl_target = xl[xl["Date"].dt.strftime("%Y-%m-%d") == TARGET]
    xl_latest = xl["Date"].max().date()
    print(f"  {TARGET} prop rows : {len(xl_target)}")
    print(f"  Latest date      : {xl_latest}")
    if len(xl_target):
        games = xl_target[["Game", "Game_Time_ET"]].drop_duplicates("Game")
        print(f"  {TARGET} games:")
        for _, r in games.iterrows():
            print(f"    {r['Game']}  @  {r['Game_Time_ET']}")

# ── 5. Summary & recommended action ───────────────────────────
print(f"\n{SEP}")
print("  DIAGNOSIS SUMMARY")
print(SEP)

has_today   = len([p for p in (today if FILE_TODAY.exists() else []) if p["date"]==TARGET]) > 0
has_season  = len(season_target) > 0
has_csv     = len(gl_played) > 0
has_excel   = len(xl_target) > 0

state = {
    f"{TARGET} in today.json  ": ("YES (" + str(len([p for p in (today if FILE_TODAY.exists() else []) if p['date']==TARGET])) + " plays)") if has_today else "NO",
    f"{TARGET} in season JSON ": ("YES (" + str(len(season_target)) + " plays)") if has_season else "NO",
    f"{TARGET} in game log CSV": ("YES (" + str(len(gl_played)) + " played rows)") if has_csv else "NO",
    f"{TARGET} in Excel props ": ("YES (" + str(len(xl_target)) + " props)") if has_excel else "NO",
}
for k, v in state.items():
    symbol = "✓" if "YES" in v else "✗"
    print(f"  {symbol}  {k}: {v}")

print()

# Determine recovery path
if has_season and has_csv:
    ungraded = [p for p in season_target if not p.get("result") or p["result"] == ""]
    graded   = [p for p in season_target if p.get("result") in ("WIN","LOSS","DNP")]
    print(f"  ✓ FULLY RECOVERABLE")
    print(f"    Plays in season JSON : {len(season_target)} ({len(ungraded)} ungraded, {len(graded)} graded)")
    print(f"    Results in CSV       : {len(gl_played)} played rows")
    print(f"")
    print(f"  ▶ Run: python3 run.py grade-csv --date {TARGET} --no-retrain")

elif has_season and not has_csv:
    print(f"  ⚠ PARTIAL — plays exist in season JSON but NO results in game log CSV")
    print(f"    The game log has not been updated with {TARGET} box scores yet.")
    print(f"")
    print(f"  ▶ Option 1 (NBA API):  python3 run.py grade --date {TARGET} --no-retrain")
    print(f"  ▶ Option 2 (if API still fails): manually add {TARGET} rows to")
    print(f"    source-files/nba_gamelogs_2025_26.csv then run:")
    print(f"    python3 run.py grade-csv --date {TARGET} --no-retrain")

elif not has_season and has_excel:
    print(f"  ✗ PLAYS NOT IN SEASON JSON — need to re-run batch predict for {TARGET}")
    print(f"    The {TARGET} props exist in Excel ({len(xl_target)} rows) but were never")
    print(f"    scored and written to season_2025_26.json.")
    print(f"")
    print(f"  ▶ Step 1: python3 run.py predict 2  (re-score {TARGET} props)")
    print(f"    Note: today.json date will need to be {TARGET}, or edit batch_predict.py")
    print(f"    to use a fixed date. Alternatively regenerate all season JSON which")
    print(f"    covers all Excel dates including {TARGET}:")
    print(f"  ▶ Step 1 (easier): python3 generate_season_json.py --date {TARGET}")
    print(f"  ▶ Step 2:          python3 run.py grade --date {TARGET} --no-retrain")

elif not has_season and not has_excel:
    print(f"  ✗ UNRECOVERABLE — no props data found for {TARGET}")
    print(f"    The Excel file does not contain {TARGET} props.")

else:
    print(f"  ? UNKNOWN STATE — check files manually")

print()
print(SEP)
print("  Run 'python3 diagnose_apr2.py --date YYYY-MM-DD' for any date")
print(SEP)
