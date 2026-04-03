"""
PropEdge V14.0 — scheduler.py
Smart macOS launchd scheduler.
- Weekday:  B0=08:00, B1=08:30, B2=18:30, B3=20:30, B4=23:30 (UK)
- Weekend:  B0=08:00 fixed. B1/B2/B3/B4 shift relative to first NBA tip-off
            detected from The Odds API (or falls back to weekday times).

Usage:
  python3 scheduler.py install   — write and load all launchd plists
  python3 scheduler.py uninstall — unload and remove all plists
  python3 scheduler.py status    — show all agent states
  python3 scheduler.py next      — print next scheduled run times
  python3 scheduler.py weekend-check — detect weekend tip-off and adjust B1-B4

This script is also called daily by a cron-like 06:00 job to recalculate
weekend schedules. On Monday–Friday the schedule is fixed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.resolve()
PLIST_DIR  = Path.home() / "Library" / "LaunchAgents"
PYTHON     = sys.executable
LOG_DIR    = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

_UK = ZoneInfo("Europe/London")
_ET = ZoneInfo("America/New_York")

# Launchd plist labels
AGENTS = {
    "b0": "com.propedge.v14.batch0",   # grade + retrain
    "b1": "com.propedge.v14.batch1",   # morning prop scan
    "b2": "com.propedge.v14.batch2",   # main pre-game
    "b3": "com.propedge.v14.batch3",   # evening top-up
    "b4": "com.propedge.v14.batch4",   # late night / west-coast tip
    "wk": "com.propedge.v14.weekend",  # weekend reschedule runner
    "db": "com.propedge.v14.daily",    # daily schedule recalc (05:55 UK)
}

# Fixed weekday times (UK local — launchd uses local time)
WEEKDAY_TIMES = {
    "b0": (8,  0),   # 08:00 — grade + retrain
    "b1": (8,  30),  # 08:30 — morning scan
    "b2": (18, 30),  # 18:30 — main prediction run
    "b3": (20, 30),  # 20:30 — top-up
    "b4": (23, 30),  # 23:30 — late West-Coast tip
}

# Weekend offsets relative to first tip (ET → converted to UK)
# Negative = before tip, positive = after tip
WEEKEND_OFFSETS_MINS = {
    "b1": -90,   # 90 min before first tip
    "b2": -60,   # 60 min before first tip
    "b3": +30,   # 30 min after first tip
    "b4": +180,  # 3 hours after first tip (covers late West games)
}

WEEKEND_FLOOR = {     # never earlier than these times (UK) on weekends
    "b1": (10, 30),
    "b2": (16, 0),
    "b3": (18, 0),
    "b4": (21, 0),
}

WEEKEND_CEIL = {      # never later than these times (UK) on weekends
    "b1": (14, 0),
    "b2": (19, 0),
    "b3": (22, 0),
    "b4": (23, 59),
}

ODDS_API_KEY = None   # loaded from config at runtime


# ─────────────────────────────────────────────────────────────────────────────
# WEEKEND TIP-OFF DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    global ODDS_API_KEY
    if ODDS_API_KEY is None:
        try:
            sys.path.insert(0, str(ROOT))
            from config import ODDS_API_KEY as KEY
            ODDS_API_KEY = KEY
        except Exception:
            ODDS_API_KEY = ""
    return ODDS_API_KEY


def fetch_first_tip_et(date_str: str) -> datetime | None:
    """
    Query The Odds API for NBA events on date_str.
    Return the earliest tip-off as a timezone-aware datetime in ET.
    Returns None on failure.
    """
    key = _get_api_key()
    if not key:
        return None

    from config import et_window
    fr_utc, to_utc = et_window(date_str)
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/events",
            params={
                "apiKey": key,
                "commenceTimeFrom": fr_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "commenceTimeTo":   to_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            timeout=10,
        )
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        print(f"  [scheduler] Tip-off API error: {e}")
        return None

    if not events:
        return None

    earliest = None
    for ev in events:
        ts = ev.get("commence_time", "")
        if not ts:
            continue
        try:
            from datetime import timezone
            dt_utc = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            dt_et  = dt_utc.astimezone(_ET)
            if earliest is None or dt_et < earliest:
                earliest = dt_et
        except Exception:
            continue

    return earliest


def compute_weekend_times(date_str: str) -> dict[str, tuple[int, int]]:
    """
    Compute B1-B4 UK times for a weekend date based on first tip-off.
    Falls back to weekday times if tip detection fails.
    """
    first_tip_et = fetch_first_tip_et(date_str)

    if first_tip_et is None:
        print("  [scheduler] Tip-off detection failed — using weekday fallback times.")
        return {k: WEEKDAY_TIMES[k] for k in ("b1","b2","b3","b4")}

    first_tip_uk = first_tip_et.astimezone(_UK)
    print(f"  [scheduler] First tip detected: {first_tip_et.strftime('%H:%M ET')} "
          f"= {first_tip_uk.strftime('%H:%M UK')}")

    result: dict[str, tuple[int, int]] = {}
    for batch, offset_mins in WEEKEND_OFFSETS_MINS.items():
        target = first_tip_uk + timedelta(minutes=offset_mins)
        hour, minute = target.hour, target.minute

        # Apply floor
        fl_h, fl_m = WEEKEND_FLOOR[batch]
        if (hour, minute) < (fl_h, fl_m):
            hour, minute = fl_h, fl_m

        # Apply ceiling
        ce_h, ce_m = WEEKEND_CEIL[batch]
        if (hour, minute) > (ce_h, ce_m):
            hour, minute = ce_h, ce_m

        result[batch] = (hour, minute)
        print(f"    {batch.upper()}: {hour:02d}:{minute:02d} UK (offset {offset_mins:+d}min from tip)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PLIST GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _plist_content(label: str, script: str, hour: int, minute: int,
                   log_name: str, args: list[str] | None = None) -> str:
    """Generate a launchd plist XML string."""
    prog_args = f"        <string>{PYTHON}</string>\n        <string>{ROOT / script}</string>"
    if args:
        for a in args:
            prog_args += f"\n        <string>{a}</string>"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
{prog_args}
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>   <integer>{hour}</integer>
        <key>Minute</key> <integer>{minute}</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>{LOG_DIR / log_name}.log</string>
    <key>StandardErrorPath</key>
    <string>{LOG_DIR / log_name}_err.log</string>

    <key>RunAtLoad</key>  <false/>
    <key>WorkingDirectory</key>
    <string>{ROOT}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:{Path(PYTHON).parent}</string>
        <key>HOME</key>
        <string>{Path.home()}</string>
    </dict>
</dict>
</plist>
"""


def _weekend_runner_plist(label: str) -> str:
    """
    Plist for the daily schedule recalculator (runs at 05:55 UK every day).
    On Mon-Fri it's a no-op; on Sat/Sun it rewrites B1-B4 plists.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{PYTHON}</string>
        <string>{ROOT / "scheduler.py"}</string>
        <string>daily-recalc</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>   <integer>5</integer>
        <key>Minute</key> <integer>55</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>{LOG_DIR / "scheduler"}.log</string>
    <key>StandardErrorPath</key>
    <string>{LOG_DIR / "scheduler"}_err.log</string>

    <key>RunAtLoad</key>  <false/>
    <key>WorkingDirectory</key>
    <string>{ROOT}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:{Path(PYTHON).parent}</string>
        <key>HOME</key>
        <string>{Path.home()}</string>
    </dict>
</dict>
</plist>
"""


# ─────────────────────────────────────────────────────────────────────────────
# LAUNCHCTL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _launchctl(cmd: list[str]) -> bool:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def _load_plist(path: Path) -> None:
    _launchctl(["launchctl", "unload", str(path)])  # unload first if already loaded
    if _launchctl(["launchctl", "load", str(path)]):
        print(f"  ✓ Loaded: {path.name}")
    else:
        print(f"  ✗ Failed to load: {path.name}")


def _unload_plist(path: Path) -> None:
    if _launchctl(["launchctl", "unload", str(path)]):
        print(f"  ✓ Unloaded: {path.name}")
    if path.exists():
        path.unlink()
        print(f"  ✓ Deleted: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# INSTALL
# ─────────────────────────────────────────────────────────────────────────────

def install(times: dict[str, tuple[int, int]] | None = None) -> None:
    """Write all plists with current schedule and load via launchctl."""
    if times is None:
        times = WEEKDAY_TIMES

    PLIST_DIR.mkdir(parents=True, exist_ok=True)
    plists = {}

    # Batch 0 — always fixed at 08:00
    plists["b0"] = PLIST_DIR / f"{AGENTS['b0']}.plist"
    plists["b0"].write_text(_plist_content(
        AGENTS["b0"], "batch0_grade.py", *WEEKDAY_TIMES["b0"], "batch0"))

    # Batch 1–4
    for bk in ("b1","b2","b3","b4"):
        plists[bk] = PLIST_DIR / f"{AGENTS[bk]}.plist"
        script = "batch_predict.py"
        arg    = str({"b1":"1","b2":"2","b3":"3","b4":"4"}[bk])
        plists[bk].write_text(_plist_content(
            AGENTS[bk], script, *times[bk], bk, args=[arg]))

    # Daily recalculator
    plists["db"] = PLIST_DIR / f"{AGENTS['db']}.plist"
    plists["db"].write_text(_weekend_runner_plist(AGENTS["db"]))

    print("\n  Loading launchd agents...")
    for key, path in plists.items():
        _load_plist(path)

    print(f"\n  Schedule installed:")
    print(f"    B0 Grade:       08:00 UK (fixed)")
    for bk in ("b1","b2","b3","b4"):
        h, m = times[bk]
        print(f"    {bk.upper()} Predict:  {h:02d}:{m:02d} UK")
    print(f"    Daily recalc:   05:55 UK")


def uninstall() -> None:
    """Unload and remove all PropEdge V14 launchd agents."""
    for label in AGENTS.values():
        path = PLIST_DIR / f"{label}.plist"
        if path.exists():
            _unload_plist(path)
    print("  All V14 agents removed.")


# ─────────────────────────────────────────────────────────────────────────────
# STATUS
# ─────────────────────────────────────────────────────────────────────────────

def status() -> None:
    print(f"\n  {'Agent':<40} {'Status':>10}")
    print(f"  {'─'*52}")
    for key, label in AGENTS.items():
        path = PLIST_DIR / f"{label}.plist"
        result = subprocess.run(
            ["launchctl", "list", label], capture_output=True, text=True
        )
        if result.returncode == 0:
            state = "LOADED ✓"
        elif path.exists():
            state = "NOT LOADED"
        else:
            state = "NOT INSTALLED"
        print(f"  {label:<40} {state:>12}")


# ─────────────────────────────────────────────────────────────────────────────
# DAILY RECALCULATION (called at 05:55 UK by launchd)
# ─────────────────────────────────────────────────────────────────────────────

def daily_recalc() -> None:
    """
    Called every morning at 05:55 UK by the daily agent.
    - Mon–Fri: no-op (weekday times are already set in the plists).
    - Sat–Sun: fetch first tip-off and rewrite + reload B1-B4 plists.
    """
    now_uk  = datetime.now(_UK)
    weekday = now_uk.weekday()   # 0=Mon, 5=Sat, 6=Sun
    date_str = now_uk.strftime("%Y-%m-%d")

    print(f"[daily-recalc] {date_str}  weekday={weekday}")

    if weekday not in (5, 6):
        print("  Weekday — no schedule change needed.")
        # Ensure weekday times are in place (safety reinstall)
        _reinstall_predict_plists(WEEKDAY_TIMES)
        return

    print(f"  Weekend ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][weekday]}) — computing game-relative schedule...")
    weekend_times = compute_weekend_times(date_str)
    _reinstall_predict_plists(weekend_times)
    print(f"  Weekend schedule applied for {date_str}")


def _reinstall_predict_plists(times: dict[str, tuple[int, int]]) -> None:
    """Rewrite and reload B1-B4 plists only (B0 is always fixed)."""
    for bk in ("b1","b2","b3","b4"):
        path = PLIST_DIR / f"{AGENTS[bk]}.plist"
        arg  = str({"b1":"1","b2":"2","b3":"3","b4":"4"}[bk])
        path.write_text(_plist_content(
            AGENTS[bk], "batch_predict.py", *times[bk], bk, args=[arg]))
        _launchctl(["launchctl", "unload", str(path)])
        _load_plist(path)


# ─────────────────────────────────────────────────────────────────────────────
# NEXT SCHEDULED TIMES
# ─────────────────────────────────────────────────────────────────────────────

def show_next() -> None:
    """Print next run time for each agent from current plists."""
    print(f"\n  {'Agent':<35} {'Next run (UK)':>20}")
    print(f"  {'─'*57}")
    now_uk = datetime.now(_UK)
    for key, label in AGENTS.items():
        path = PLIST_DIR / f"{label}.plist"
        if not path.exists():
            print(f"  {label:<35} {'NOT INSTALLED':>20}")
            continue
        try:
            import plistlib
            with open(path, "rb") as f:
                pl = plistlib.load(f)
            sci = pl.get("StartCalendarInterval", {})
            h = sci.get("Hour", 0)
            m = sci.get("Minute", 0)
            candidate = now_uk.replace(hour=h, minute=m, second=0, microsecond=0)
            if candidate <= now_uk:
                candidate += timedelta(days=1)
            print(f"  {label:<35} {candidate.strftime('%a %d %b  %H:%M UK'):>20}")
        except Exception as e:
            print(f"  {label:<35} {'ERROR: '+str(e):>20}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "install":
        print("\n  Installing PropEdge V14 launchd agents (weekday schedule)...")
        install()

    elif cmd == "uninstall":
        print("\n  Uninstalling PropEdge V14 agents...")
        uninstall()

    elif cmd == "reinstall":
        print("\n  Reinstalling...")
        uninstall()
        install()

    elif cmd == "status":
        status()

    elif cmd == "next":
        show_next()

    elif cmd == "daily-recalc":
        daily_recalc()

    elif cmd == "weekend-check":
        date_str = sys.argv[2] if len(sys.argv) > 2 else datetime.now(_UK).strftime("%Y-%m-%d")
        print(f"\n  Weekend schedule preview for {date_str}:")
        times = compute_weekend_times(date_str)
        for bk, (h, m) in times.items():
            print(f"    {bk.upper()}: {h:02d}:{m:02d} UK")

    elif cmd == "help" or not cmd:
        print("""
PropEdge V14.0 — Scheduler

Commands:
  python3 scheduler.py install          Install all launchd agents (weekday times)
  python3 scheduler.py uninstall        Remove all agents
  python3 scheduler.py reinstall        Remove + reinstall
  python3 scheduler.py status           Show all agent states
  python3 scheduler.py next             Print next run times
  python3 scheduler.py daily-recalc     Run the daily schedule recalculator
  python3 scheduler.py weekend-check    Preview weekend schedule (no install)
  python3 scheduler.py weekend-check YYYY-MM-DD  (for specific date)

Schedule:
  Weekday (Mon-Fri):
    B0  08:00 UK — Grade yesterday + retrain
    B1  08:30 UK — Morning prop scan
    B2  18:30 UK — Main prediction run
    B3  20:30 UK — Top-up run
    B4  23:30 UK — Late West-Coast tip run

  Weekend (Sat-Sun):
    B0  08:00 UK — Grade + retrain (FIXED — never changes)
    B1  90 min before first NBA tip (UK time)
    B2  60 min before first NBA tip (UK time)
    B3  30 min after  first NBA tip (UK time)
    B4  3 hr after    first NBA tip (UK time)
    (floor/ceiling limits apply — see WEEKEND_FLOOR/CEIL in scheduler.py)

    Weekend times are recalculated daily at 05:55 UK using The Odds API.
""")

    else:
        print(f"  Unknown command: {cmd}. Run `python3 scheduler.py help`")


if __name__ == "__main__":
    main()
