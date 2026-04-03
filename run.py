"""
PropEdge V14.0 — run.py
Master CLI. Install agents, generate season data, manual batch runs.

Usage:
  python3 run.py install              — Install launchd scheduler
  python3 run.py uninstall            — Remove scheduler
  python3 run.py status               — Show agent states + next run times
  python3 run.py generate             — Train models + build season JSONs (first-time setup)
  python3 run.py grade                — Manual Batch 0 (grade yesterday + retrain)
  python3 run.py grade --date DATE    — Grade a specific date e.g. 2026-04-02
  python3 run.py grade --no-retrain   — Grade only, skip model retraining
  python3 run.py predict [1-4]        — Manual prediction run (default: 2)
  python3 run.py retrain              — Retrain models only (no grading)
  python3 run.py dvp                  — Rebuild DVP rankings only
  python3 run.py h2h                  — Rebuild H2H database only
  python3 run.py weekend              — Preview this weekend's schedule
  python3 run.py check                — Run data integrity checks
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import VERSION, FILE_CLF, FILE_TODAY, FILE_SEASON_2526


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
    _show_model_status()
    _show_credit_info()


def _show_model_status():
    print(f"\n  Model files:")
    from config import FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST, FILE_DVP, FILE_H2H
    files = {
        "direction_classifier.pkl": FILE_CLF,
        "projection_model.pkl":     FILE_REG,
        "calibrator.pkl":           FILE_CAL,
        "player_trust.json":        FILE_TRUST,
        "dvp_rankings.json":        FILE_DVP,
        "h2h_database.csv":         FILE_H2H,
    }
    for name, path in files.items():
        if path.exists():
            size = path.stat().st_size
            mtime = __import__("datetime").datetime.fromtimestamp(path.stat().st_mtime)
            print(f"    ✓ {name:<35} {size/1024:>8.0f}KB  {mtime.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"    ✗ {name:<35} MISSING")


def _show_credit_info():
    from config import ODDS_API_KEY, ODDS_BASE_URL
    import requests
    try:
        r = requests.get(f"{ODDS_BASE_URL}/sports", params={"apiKey": ODDS_API_KEY}, timeout=5)
        remaining = r.headers.get("x-requests-remaining", "?")
        used      = r.headers.get("x-requests-used", "?")
        print(f"\n  Odds API: {remaining} credits remaining (used: {used})")
    except Exception:
        print(f"\n  Odds API: unable to check credits")


def cmd_generate():
    """First-time setup: train models and build season JSONs."""
    print(f"\n  PropEdge {VERSION} — First-Time Generate")
    print("  This will train models and build season JSONs (~5-8 min)\n")

    # DVP + H2H first (models need them)
    from dvp_updater import compute_and_save_dvp
    from h2h_builder import build_h2h
    from config import FILE_GL_2425, FILE_GL_2526, FILE_H2H, FILE_DVP
    print("  [1/3] Building DVP rankings...")
    compute_and_save_dvp(FILE_GL_2526, FILE_DVP)
    print("  [2/3] Building H2H database...")
    build_h2h(FILE_GL_2425, FILE_GL_2526, FILE_H2H)
    print("  [3/3] Training V14 models...")
    from model_trainer import train_and_save
    from config import FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST
    train_and_save(FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST)

    # Initialise empty season JSONs if not present
    from config import FILE_SEASON_2526, FILE_SEASON_2425, FILE_TODAY, clean_json
    import json
    for path in (FILE_SEASON_2425, FILE_SEASON_2526, FILE_TODAY):
        if not path.exists():
            path.write_text("[]")
            print(f"  Created empty: {path.name}")

    print("\n  ✓ Generate complete. Run `python3 run.py predict` to test.")


def cmd_grade():
    """Manual Batch 0. Passes --date and --no-retrain through to batch0_grade."""
    import batch0_grade
    batch0_grade.main()


def cmd_predict(batch_num: int = 2):
    """Manual prediction batch."""
    import importlib, sys
    sys.argv = ["batch_predict.py", str(batch_num)]
    import batch_predict
    importlib.reload(batch_predict)
    batch_predict.main()


def cmd_retrain():
    """Retrain models only."""
    from model_trainer import train_and_save
    from config import FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST
    print(f"  Retraining {VERSION} models...")
    train_and_save(FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST)


def cmd_dvp():
    """Rebuild DVP only."""
    from dvp_updater import compute_and_save_dvp
    from config import FILE_GL_2526, FILE_DVP
    compute_and_save_dvp(FILE_GL_2526, FILE_DVP)


def cmd_h2h():
    """Rebuild H2H database only."""
    from h2h_builder import build_h2h
    from config import FILE_GL_2425, FILE_GL_2526, FILE_H2H
    build_h2h(FILE_GL_2425, FILE_GL_2526, FILE_H2H)


def cmd_weekend():
    """Preview this weekend's or a given date's schedule."""
    from scheduler import compute_weekend_times
    from datetime import datetime
    from zoneinfo import ZoneInfo
    date_arg = sys.argv[2] if len(sys.argv) > 2 else datetime.now(ZoneInfo("Europe/London")).strftime("%Y-%m-%d")
    print(f"\n  Weekend schedule preview for {date_arg}:")
    times = compute_weekend_times(date_arg)
    for bk, (h, m) in times.items():
        print(f"    {bk.upper()}: {h:02d}:{m:02d} UK")


def cmd_check():
    """Quick data integrity checks."""
    import pandas as pd
    from config import FILE_GL_2526, FILE_H2H, FILE_TODAY, FILE_SEASON_2526, FILE_AUDIT

    print(f"\n  Data Integrity Checks — PropEdge {VERSION}")
    print(f"  {'─'*50}")

    # Game log
    try:
        gl = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
        print(f"  GL2526: {len(gl):,} rows | {gl['PLAYER_NAME'].nunique()} players | "
              f"latest: {gl['GAME_DATE'].max().date()}")
    except Exception as e:
        print(f"  GL2526: ERROR — {e}")

    # H2H
    try:
        h2h = pd.read_csv(FILE_H2H)
        print(f"  H2H:    {len(h2h):,} matchup pairs")
    except Exception as e:
        print(f"  H2H:    ERROR — {e}")

    # Today JSON
    import json
    try:
        with open(FILE_TODAY) as f: today = json.load(f)
        graded   = [p for p in today if p.get("result") in ("WIN","LOSS","DNP")]
        ungraded = [p for p in today if p not in graded]
        t1 = [p for p in today if p.get("tier") == 1]
        t2 = [p for p in today if p.get("tierLabel") == "T2"]
        print(f"  Today:  {len(today)} plays | T1:{len(t1)} T2:{len(t2)} | "
              f"graded:{len(graded)} pending:{len(ungraded)}")
    except Exception as e:
        print(f"  Today:  ERROR — {e}")

    # Season JSON
    try:
        with open(FILE_SEASON_2526) as f: season = json.load(f)
        wins   = sum(1 for p in season if p.get("result") == "WIN")
        losses = sum(1 for p in season if p.get("result") == "LOSS")
        t1_s   = [p for p in season if p.get("tier") == 1 and p.get("result") in ("WIN","LOSS")]
        t1_acc = sum(1 for p in t1_s if p.get("result")=="WIN") / max(len(t1_s),1)
        print(f"  Season: {len(season):,} plays | W:{wins} L:{losses} | T1 acc:{t1_acc:.1%}")
    except Exception as e:
        print(f"  Season: ERROR — {e}")

    # Audit log
    try:
        audit = pd.read_csv(FILE_AUDIT)
        alerts = audit[audit["event"].str.contains("ALERT|FAIL", na=False)]
        print(f"  Audit:  {len(audit):,} events | {len(alerts)} warnings")
        if not alerts.empty:
            for _, row in alerts.tail(3).iterrows():
                print(f"    ⚠ [{row['ts']}] {row['event']}: {row['detail']}")
    except Exception as e:
        print(f"  Audit:  {e}")

    # Model freshness
    if FILE_CLF.exists():
        import datetime as dt_mod
        age_h = (dt_mod.datetime.now() - dt_mod.datetime.fromtimestamp(FILE_CLF.stat().st_mtime)).total_seconds() / 3600
        print(f"  Models: last retrained {age_h:.0f}h ago")
    else:
        print(f"  Models: NOT FOUND — run `python3 run.py generate`")

    print()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    print(f"\n  PropEdge {VERSION}  —  {cmd}")

    dispatch = {
        "install":   cmd_install,
        "uninstall": cmd_uninstall,
        "status":    cmd_status,
        "generate":  cmd_generate,
        "grade":     cmd_grade,
        "retrain":   cmd_retrain,
        "dvp":       cmd_dvp,
        "h2h":       cmd_h2h,
        "weekend":   cmd_weekend,
        "check":     cmd_check,
    }

    if cmd == "predict":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        cmd_predict(n)
    elif cmd in dispatch:
        dispatch[cmd]()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
