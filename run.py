"""
PropEdge V14.0 — run.py
Master CLI.

FRESH START:
  python3 run.py setup     — Build DVP + H2H + train models from source CSVs.
                             Does NOT touch data JSONs. ~5-8 min.
  python3 run.py start     — Auto-grade yesterday (from CSV) then predict today.
                             Use this after setup on a fresh machine.

DAILY (automated via scheduler):
  Batch 0 — python3 run.py grade        (08:00 UK — grade yesterday via NBA API)
  Batch 1 — python3 run.py predict 1   (pre-game early lines)
  Batch 2 — python3 run.py predict 2   (main pre-game batch)
  Batch 3 — python3 run.py predict 3   (in-game update)
  Batch 4 — python3 run.py predict 4   (final)

MANUAL OVERRIDES:
  python3 run.py grade --date 2026-04-02           Grade a specific date via NBA API
  python3 run.py grade --no-retrain                Grade only, skip model retrain
  python3 run.py grade-csv --date 2026-04-02       Grade from game log CSV (API bypass)
  python3 run.py grade-csv --date 2026-04-02 --no-retrain

UTILITIES:
  python3 run.py retrain   — Retrain models from source CSVs only
  python3 run.py dvp       — Rebuild DVP rankings
  python3 run.py h2h       — Rebuild H2H database
  python3 run.py check     — Data integrity report
  python3 run.py status    — Scheduler + model status
  python3 run.py install   — Install launchd scheduler (macOS)
  python3 run.py uninstall — Remove scheduler
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    VERSION, FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST,
    FILE_GL_2425, FILE_GL_2526, FILE_H2H, FILE_DVP,
    FILE_TODAY, FILE_SEASON_2526, FILE_PROPS, clean_json,
)


# ─────────────────────────────────────────────────────────────────────────────
# SETUP — train models from source CSVs, no JSON generation
# ─────────────────────────────────────────────────────────────────────────────

def cmd_setup():
    """
    Build DVP + H2H + train V14 models from source CSV files.
    Does NOT generate or wipe any data JSONs.
    Run once on a fresh machine before using grade/predict.
    """
    print(f"\n  PropEdge {VERSION} — Setup")
    print("  Builds models from source CSVs. Data JSONs are untouched.\n")

    # Validate source files exist
    missing = [f for f in (FILE_GL_2425, FILE_GL_2526, FILE_PROPS) if not f.exists()]
    if missing:
        print("  ✗ Missing source files:")
        for f in missing:
            print(f"    {f}")
        print("  Add these to source-files/ and retry.")
        return

    import pandas as pd
    gl26 = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
    print(f"  Game log 2025-26 : {len(gl26):,} rows | "
          f"latest: {gl26['GAME_DATE'].max().date()}")
    gl25 = pd.read_csv(FILE_GL_2425, parse_dates=["GAME_DATE"])
    print(f"  Game log 2024-25 : {len(gl25):,} rows")

    from dvp_updater import compute_and_save_dvp
    from h2h_builder import build_h2h
    from model_trainer import train_and_save

    print("\n  [1/3] Building DVP rankings...")
    compute_and_save_dvp(FILE_GL_2526, FILE_DVP)

    print("  [2/3] Building H2H database...")
    build_h2h(FILE_GL_2425, FILE_GL_2526, FILE_H2H)

    print("  [3/3] Training V14 models (5-8 min)...")
    train_and_save(FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST)

    # Create empty data JSONs only if they don't already exist
    for path in (FILE_TODAY, FILE_SEASON_2526):
        if not path.exists():
            path.write_text("[]")
            print(f"  Created empty: {path.name}")
        else:
            print(f"  Kept existing: {path.name}")

    print("\n  ✓ Setup complete.")
    print("  Next: python3 run.py start")
    print("        (auto-grades yesterday from CSV then predicts today)\n")


# ─────────────────────────────────────────────────────────────────────────────
# START — auto-grade yesterday + predict today (fresh run entry point)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_start():
    """
    Auto-grade yesterday from game log CSV, then predict today's props.
    The clean entry point after setup. Does not rely on NBA API.
    """
    from datetime import datetime, timedelta
    from config import get_uk

    uk_now    = datetime.now(get_uk())
    today_str = uk_now.strftime("%Y-%m-%d")
    yest_str  = (uk_now - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\n  PropEdge {VERSION} — Start")
    print(f"  UK date : {today_str}")
    print(f"  Grading : {yest_str}")
    print(f"  Predict : {today_str}\n")

    # Check models exist
    if not FILE_CLF.exists():
        print("  ✗ Models not found — run `python3 run.py setup` first.")
        return

    # ── Step 1: grade yesterday from CSV ──────────────────────────────────
    import pandas as pd
    gl = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
    yest_gl   = gl[gl["GAME_DATE"].dt.strftime("%Y-%m-%d") == yest_str]
    yest_play = yest_gl[
        (yest_gl["DNP"].fillna(0) == 0) & (yest_gl["MIN_NUM"].fillna(0) > 0)
    ]

    if len(yest_play) == 0:
        print(f"  ⚠ No game log rows for {yest_str} — skipping grade step.")
        print(f"    If games were played, update your game log CSV first.")
    else:
        print(f"  [1/2] Grading {yest_str} from CSV ({len(yest_play)} played rows)...")
        _grade_from_csv(yest_str, no_retrain=True)

    # ── Step 2: predict today ──────────────────────────────────────────────
    xl = pd.read_excel(FILE_PROPS, sheet_name="Player_Points_Props")
    xl["Date"] = pd.to_datetime(xl["Date"])
    today_xl   = xl[xl["Date"].dt.strftime("%Y-%m-%d") == today_str]

    if len(today_xl) == 0:
        print(f"\n  ⚠ No props in Excel for {today_str}.")
        print(f"    Excel latest date: {xl['Date'].max().date()}")
        print(f"    Add today's props to the Excel file then run:")
        print(f"    python3 run.py predict 2")
    else:
        print(f"\n  [2/2] Predicting {len(today_xl)} props for {today_str}...")
        cmd_predict(2)

    print(f"\n  ✓ Start complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# GRADE (NBA API)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_grade():
    """Grade yesterday via NBA API. Passes --date and --no-retrain through."""
    import batch0_grade
    batch0_grade.main()


# ─────────────────────────────────────────────────────────────────────────────
# GRADE-CSV (game log bypass)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_grade_from_csv():
    """
    Grade a specific date from the game log CSV — no NBA API needed.
    Usage: python3 run.py grade-csv --date 2026-04-02 [--no-retrain]
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",        required=True)
    parser.add_argument("--no-retrain",  action="store_true")
    args, _ = parser.parse_known_args()

    print(f"\n  Grading {args.date} from game log CSV...")
    graded = _grade_from_csv(args.date, no_retrain=args.no_retrain)

    from batch_predict import git_push
    git_push(f"B0-csv: grade {args.date}")
    print(f"  ✓ CSV grade complete — {graded['wins']}W / {graded['losses']}L / {graded['dnps']} DNP\n")


def _grade_from_csv(grade_date: str, no_retrain: bool = False) -> dict:
    """
    Core grading logic using game log CSV.
    Matches plays in today.json and season JSON against CSV results.
    Returns {wins, losses, dnps}.
    """
    import pandas as pd
    from reasoning_engine import generate_post_match_reason

    gl = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
    day_gl  = gl[gl["GAME_DATE"].dt.strftime("%Y-%m-%d") == grade_date]
    played  = day_gl[(day_gl["DNP"].fillna(0) == 0) & (day_gl["MIN_NUM"].fillna(0) > 0)]

    if len(played) == 0:
        print(f"  ⚠ No played rows in CSV for {grade_date}")
        return {"wins": 0, "losses": 0, "dnps": 0}

    # Build a results map: player_name → row  (last entry wins if dupes)
    results_map     = {}
    players_in_box  = set()
    for _, r in played.iterrows():
        name = r["PLAYER_NAME"]
        players_in_box.add(name)
        results_map[name] = r.to_dict()

    print(f"  CSV: {len(played)} played rows, {len(players_in_box)} unique players")

    wins = losses = dnps = 0

    for path in (FILE_TODAY, FILE_SEASON_2526):
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        changed = False

        for play in data:
            if play.get("date") != grade_date:
                continue
            if play.get("result") in ("WIN", "LOSS", "DNP"):
                continue   # immutable

            pname    = play.get("player", "")
            line     = float(play.get("line", 20))
            dr       = str(play.get("direction", ""))
            is_over  = "OVER"  in dr.upper() and "LEAN" not in dr.upper()
            is_under = "UNDER" in dr.upper() and "LEAN" not in dr.upper()

            box = results_map.get(pname)

            if pname not in players_in_box or box is None:
                play["result"]         = "DNP"
                play["actualPts"]      = None
                play["actualMin"]      = 0
                post, lt               = generate_post_match_reason(play)
                play["postMatchReason"] = post
                play["lossType"]       = "DNP"
                dnps += 1
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
                    "actual_pts":    actual,
                    "actual_min":    actual_min,
                    "actual_fga":    box.get("FGA", 0),
                    "actual_fgm":    box.get("FGM", 0),
                    "actual_fg_pct": box.get("FGM", 0) / max(box.get("FGA", 1), 1),
                }
                post, lt               = generate_post_match_reason(play, box_data)
                play["postMatchReason"] = post
                play["lossType"]       = lt

            changed = True

        if changed:
            with open(path, "w") as f:
                json.dump(clean_json(data), f, indent=2)

    print(f"  Graded: W:{wins}  L:{losses}  DNP:{dnps}")

    if not no_retrain:
        from model_trainer import train_and_save
        print("  Retraining models...")
        train_and_save(FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST)

    return {"wins": wins, "losses": losses, "dnps": dnps}


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def cmd_predict(batch_num: int = 2):
    """Run a prediction batch (1-4). Default: 2."""
    import importlib
    sys.argv = ["batch_predict.py", str(batch_num)]
    import batch_predict
    importlib.reload(batch_predict)
    batch_predict.main()


# ─────────────────────────────────────────────────────────────────────────────
# RETRAIN
# ─────────────────────────────────────────────────────────────────────────────

def cmd_retrain():
    """Retrain V14 models from source CSVs only."""
    from model_trainer import train_and_save
    print(f"  Retraining {VERSION} models from source CSVs...")
    train_and_save(FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST)
    print("  ✓ Retrain complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# DVP / H2H
# ─────────────────────────────────────────────────────────────────────────────

def cmd_dvp():
    from dvp_updater import compute_and_save_dvp
    compute_and_save_dvp(FILE_GL_2526, FILE_DVP)


def cmd_h2h():
    from h2h_builder import build_h2h
    build_h2h(FILE_GL_2425, FILE_GL_2526, FILE_H2H)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK
# ─────────────────────────────────────────────────────────────────────────────

def cmd_check():
    """Data integrity report."""
    import pandas as pd
    from config import FILE_AUDIT

    print(f"\n  PropEdge {VERSION} — Data Integrity Check")
    print(f"  {'─'*55}")

    # Source files
    for label, path in [("GL 2025-26", FILE_GL_2526), ("GL 2024-25", FILE_GL_2425),
                        ("H2H DB",     FILE_H2H),     ("Excel props", FILE_PROPS)]:
        if path.exists():
            size = path.stat().st_size / 1024
            if path.suffix == ".csv":
                try:
                    df = pd.read_csv(path, parse_dates=["GAME_DATE"] if "gamelogs" in path.name else None)
                    extra = f" | {len(df):,} rows"
                    if "GAME_DATE" in df.columns:
                        extra += f" | latest: {df['GAME_DATE'].max().date()}"
                except Exception:
                    extra = ""
            elif path.suffix == ".xlsx":
                try:
                    df = pd.read_excel(path, sheet_name="Player_Points_Props")
                    df["Date"] = pd.to_datetime(df["Date"])
                    extra = f" | {len(df):,} rows | latest: {df['Date'].max().date()}"
                except Exception:
                    extra = ""
            else:
                extra = ""
            print(f"  ✓ {label:<15} {size:>8.0f} KB{extra}")
        else:
            print(f"  ✗ {label:<15} MISSING — {path}")

    print(f"  {'─'*55}")

    # Models
    for label, path in [("Classifier",  FILE_CLF), ("Regressor", FILE_REG),
                        ("Calibrator",  FILE_CAL), ("Trust",     FILE_TRUST)]:
        if path.exists():
            import datetime as _dt
            age_h = (_dt.datetime.now() -
                     _dt.datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 3600
            print(f"  ✓ Model {label:<12} {path.stat().st_size/1024:>8.0f} KB  ({age_h:.0f}h ago)")
        else:
            print(f"  ✗ Model {label:<12} MISSING — run `python3 run.py setup`")

    print(f"  {'─'*55}")

    # Data JSONs
    for label, path in [("today.json", FILE_TODAY), ("season_2025_26", FILE_SEASON_2526)]:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            dates   = sorted(set(p["date"] for p in data))
            graded  = sum(1 for p in data if p.get("result") in ("WIN","LOSS"))
            ungrad  = sum(1 for p in data if not p.get("result"))
            dnps    = sum(1 for p in data if p.get("result") == "DNP")
            wins    = sum(1 for p in data if p.get("result") == "WIN")
            hr      = f"{wins/graded*100:.1f}%" if graded else "—"
            print(f"  ✓ {label:<18} {len(data):>6} plays | "
                  f"graded:{graded} ungraded:{ungrad} DNP:{dnps} | "
                  f"HR:{hr}")
            if dates:
                print(f"    dates: {dates[-3:]}")
        else:
            print(f"  ✗ {label:<18} MISSING")

    print(f"  {'─'*55}")

    # Audit log
    try:
        audit = pd.read_csv(FILE_AUDIT)
        alerts = audit[audit["event"].str.contains("FAIL|ALERT|ABORT", na=False)]
        print(f"  Audit: {len(audit):,} events | {len(alerts)} warnings")
        for _, row in alerts.tail(3).iterrows():
            print(f"    ⚠ [{row.get('ts','')}] {row.get('event','')}: {row.get('detail','')}")
    except Exception:
        print(f"  Audit: no log yet")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def cmd_install():
    from scheduler import install
    install()


def cmd_uninstall():
    from scheduler import uninstall
    uninstall()


def cmd_status():
    from scheduler import status, show_next
    status()
    show_next()
    cmd_check()


def cmd_weekend():
    from scheduler import compute_weekend_times
    from datetime import datetime
    from zoneinfo import ZoneInfo
    date_arg = sys.argv[2] if len(sys.argv) > 2 else \
               datetime.now(ZoneInfo("Europe/London")).strftime("%Y-%m-%d")
    print(f"\n  Weekend schedule preview for {date_arg}:")
    times = compute_weekend_times(date_arg)
    for bk, (h, m) in times.items():
        print(f"    {bk.upper()}: {h:02d}:{m:02d} UK")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    print(f"\n  PropEdge {VERSION}  —  {cmd}")

    dispatch = {
        "setup":      cmd_setup,
        "start":      cmd_start,
        "grade":      cmd_grade,
        "grade-csv":  cmd_grade_from_csv,
        "retrain":    cmd_retrain,
        "dvp":        cmd_dvp,
        "h2h":        cmd_h2h,
        "check":      cmd_check,
        "status":     cmd_status,
        "install":    cmd_install,
        "uninstall":  cmd_uninstall,
        "weekend":    cmd_weekend,
        # legacy alias
        "generate":   cmd_setup,
    }

    if cmd == "predict":
        n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 2
        cmd_predict(n)
    elif cmd in dispatch:
        dispatch[cmd]()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
