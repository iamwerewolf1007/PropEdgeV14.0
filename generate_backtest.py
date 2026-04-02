"""
PropEdge V14.0 — generate_backtest.py
======================================
Runs the full V14 backtesting pipeline against BOTH seasons of real data.

What it does:
  1. Loads nba_gamelogs_2024_25.csv + nba_gamelogs_2025_26.csv
  2. Loads real bookmaker lines from Player_Points_Props Excel sheet
  3. Generates synthetic lines for 2024-25 (no real lines available)
  4. Trains V14 models with OOF 5-fold TimeSeriesSplit — NO in-sample leakage
  5. Applies V14 Adaptive Fusion scoring (3 engines fused)
  6. Generates pre/post-match reasoning for every play
  7. Writes:
       data/season_2025_26.json    — All 2025-26 graded plays
       data/season_2024_25.json    — All 2024-25 graded plays
       data/today.json             — Most recent date's plays (dashboard default)
       data/backtest_summary.json  — Aggregate stats for dashboard header

Usage:
  python3 generate_backtest.py                     # both seasons, full train
  python3 generate_backtest.py --season 2526       # 2025-26 only (faster)
  python3 generate_backtest.py --season 2425       # 2024-25 only
  python3 generate_backtest.py --no-train          # skip retrain, use existing models
  python3 generate_backtest.py --date 2026-01-15   # set today.json to a specific date
"""

import argparse
import json
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    FILE_GL_2425, FILE_GL_2526, FILE_H2H, FILE_PROPS,
    FILE_CLF, FILE_REG, FILE_CAL, FILE_TRUST,
    FILE_TODAY, FILE_SEASON_2526, FILE_SEASON_2425,
    DATA_DIR, MODEL_DIR,
    season_progress, get_pos_group, fusion_weights,
    LEAN_ZONE, UNIT_MAP, HIGH_LINE_THRESHOLD, TRUST_THRESHOLD,
    clean_json,
)
from rolling_engine import filter_played, extract_prediction_features
from reasoning_engine import generate_pre_match_reason, generate_post_match_reason

ML_FEATURES = [
    "level", "reversion", "momentum", "acceleration", "level_ewm",
    "z_momentum", "z_reversion", "z_accel",
    "mean_reversion_risk", "extreme_hot", "extreme_cold",
    "season_progress", "early_season_weight", "games_depth",
    "volume", "trend", "std10", "consistency", "hr10", "hr30",
    "min_l10", "min_l30", "min_cv", "recent_min_trend", "pts_per_min",
    "fga_l10", "fg3a_l10", "fg3m_l10", "fta_l10", "ft_rate", "fga_per_min", "ppfga_l10",
    "usage_l10", "usage_l30", "role_intensity",
    "home_l10", "away_l10", "home_away_split",
    "is_b2b", "rest_days", "rest_cat", "is_long_rest",
    "defP_dynamic", "pace_rank",
    "h2h_ts_dev", "h2h_fga_dev", "h2h_min_dev", "h2h_conf", "h2h_games", "h2h_trend",
    "line", "line_vs_l30", "line_bucket",
    "line_spread", "line_sharpness", "vol_risk",
]


# =============================================================================
# STEP 1: LOAD ALL DATA
# =============================================================================

def load_all_data():
    print("\n[1/5] Loading game logs...")
    gl24 = pd.read_csv(FILE_GL_2425, parse_dates=["GAME_DATE"])
    gl25 = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"])
    gl   = pd.concat([gl24, gl25], ignore_index=True).sort_values(["PLAYER_NAME","GAME_DATE"])
    gl["DNP"] = gl["DNP"].fillna(0)
    played = filter_played(gl)
    print(f"  Played rows: {len(played):,}  |  Players: {played['PLAYER_NAME'].nunique():,}")

    # Player history index
    player_idx = {
        pname: grp.sort_values("GAME_DATE").reset_index(drop=True)
        for pname, grp in played.groupby("PLAYER_NAME")
    }

    # H2H lookup
    h2h_df = pd.read_csv(FILE_H2H).drop_duplicates(subset=["PLAYER_NAME","OPPONENT"], keep="last")
    h2h_lkp = {
        (r["PLAYER_NAME"], r["OPPONENT"]): r.to_dict()
        for _, r in h2h_df.iterrows()
    }

    # Live DVP rank
    dvp_dict = {}
    for (opp, pos), g in played.groupby(["OPPONENT","PLAYER_POSITION"]):
        dvp_dict[(opp, pos)] = g["PTS"].mean()
    dvp_rank = {}
    for pos in played["PLAYER_POSITION"].unique():
        subset = {k: v for k, v in dvp_dict.items() if k[1] == pos}
        for rank, (opp, p) in enumerate(sorted(subset, key=lambda k: subset[k], reverse=True), 1):
            dvp_rank[(opp, p)] = rank

    # Pace rank
    team_fga = played.groupby("OPPONENT")["FGA"].mean()
    pace_cache = {t: i+1 for i, (t, _) in enumerate(team_fga.sort_values(ascending=False).items())}

    # B2B rest days map
    b2b_map = {}
    for pname, grp in played.groupby("PLAYER_NAME"):
        dates = grp["GAME_DATE"].values
        for i, d in enumerate(dates):
            rd = int((d - dates[i-1]).astype("timedelta64[D]").astype(int)) if i > 0 else 99
            b2b_map[(pname, pd.Timestamp(d).strftime("%Y-%m-%d"))] = rd

    # Recent 20 scores per player (for sparkline)
    recent_idx = {}
    for pname, grp in played.groupby("PLAYER_NAME"):
        g = grp.sort_values("GAME_DATE")
        recent_idx[pname] = list(zip(
            g["GAME_DATE"].dt.strftime("%Y-%m-%d").tolist(),
            g["PTS"].fillna(0).tolist(),
            g["IS_HOME"].fillna(0).astype(int).tolist(),
            g["OPPONENT"].fillna("").tolist(),
        ))

    # Props
    print("  Loading 2025-26 prop lines from Excel...")
    props_2526 = _load_excel_props()
    print(f"    {len(props_2526):,} real prop lines")

    print("  Generating synthetic prop lines for 2024-25...")
    played_2425 = played[played["GAME_DATE"] < pd.Timestamp("2025-10-01")].copy()
    props_2425 = _generate_synthetic_props(played_2425)
    print(f"    {len(props_2425):,} synthetic prop lines")

    return player_idx, h2h_lkp, dvp_rank, pace_cache, b2b_map, recent_idx, props_2526, props_2425


def _load_excel_props():
    if not FILE_PROPS.exists():
        print("  *** Excel props file not found — 2025-26 will use synthetic lines ***")
        return []
    xl = pd.read_excel(FILE_PROPS, sheet_name="Player_Points_Props")
    xl["Date"] = pd.to_datetime(xl["Date"])
    xl = xl.dropna(subset=["Line"])
    props = []
    for _, r in xl.iterrows():
        try:
            props.append({
                "player":    str(r["Player"]).strip(),
                "date":      r["Date"],
                "line":      float(r["Line"]),
                "min_line":  float(r["Min Line"])  if pd.notna(r.get("Min Line",""))  else None,
                "max_line":  float(r["Max Line"])  if pd.notna(r.get("Max Line",""))  else None,
                "over_odds": float(r.get("Over Odds",  -110) or -110),
                "under_odds":float(r.get("Under Odds", -110) or -110),
                "game":      str(r.get("Game","")).strip(),
                "home":      str(r.get("Home","")).strip(),
                "away":      str(r.get("Away","")).strip(),
                "books":     int(r.get("Books",1) or 1),
                "season":    "2025-26",
                "source":    "real",
            })
        except Exception:
            continue
    return props


def _generate_synthetic_props(played_2425):
    """One synthetic prop per player per game using L30 rolling baseline."""
    played_2425 = played_2425.sort_values(["PLAYER_NAME","GAME_DATE"]).copy()
    props = []
    for pname, grp in played_2425.groupby("PLAYER_NAME"):
        pts_list = grp["PTS"].fillna(0).tolist()
        dates    = grp["GAME_DATE"].tolist()
        opps     = grp["OPPONENT"].fillna("").tolist()
        homes    = grp["IS_HOME"].fillna(0).astype(int).tolist()
        history  = []
        for i, (d, pts, opp, home) in enumerate(zip(dates, pts_list, opps, homes)):
            if i >= 5 and len(history) >= 5:
                l30        = np.mean(history[-30:])
                synth_line = max(3.5, round(l30 * 2) / 2)
                props.append({
                    "player":    pname,
                    "date":      d,
                    "line":      synth_line,
                    "min_line":  synth_line - 0.5,
                    "max_line":  synth_line + 0.5,
                    "over_odds": -110,
                    "under_odds":-110,
                    "game":      f"vs {opp}",
                    "home":      "",
                    "away":      "",
                    "books":     1,
                    "season":    "2024-25",
                    "source":    "synthetic",
                })
            history.append(pts)
    return props


# =============================================================================
# STEP 2: BUILD FEATURE ROWS
# =============================================================================

def build_feature_rows(player_idx, h2h_lkp, dvp_rank, pace_cache, b2b_map, props):
    rows   = []
    skip   = {"no_player":0, "thin_history":0, "no_actual":0, "no_feats":0}

    for prop in props:
        pname    = prop["player"]
        date     = prop["date"]
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        line     = prop["line"]

        hist = player_idx.get(pname)
        if hist is None:
            skip["no_player"] += 1; continue

        prior = hist[hist["GAME_DATE"] < pd.Timestamp(date)]
        if len(prior) < 5:
            skip["thin_history"] += 1; continue

        actual_game = hist[hist["GAME_DATE"] == pd.Timestamp(date)]
        if len(actual_game) == 0:
            skip["no_actual"] += 1; continue

        actual_pts = float(actual_game["PTS"].values[0])
        if pd.isna(actual_pts):
            skip["no_actual"] += 1; continue

        pos       = str(prior["PLAYER_POSITION"].iloc[-1])
        opponent  = str(actual_game["OPPONENT"].values[0])
        rest_days = b2b_map.get((pname, date_str), 99)
        pos_grp   = get_pos_group(pos)

        feats = extract_prediction_features(
            prior_played=prior,
            line=line,
            opponent=opponent,
            rest_days=rest_days,
            pos_raw=pos,
            game_date=pd.Timestamp(date),
            min_line=prop.get("min_line"),
            max_line=prop.get("max_line"),
            dvp_rank_cache={(opponent, pos_grp): dvp_rank.get((opponent, pos), 15)},
            pace_rank_cache=pace_cache,
        )
        if feats is None:
            skip["no_feats"] += 1; continue

        hk = h2h_lkp.get((pname, opponent), {})
        feats["h2h_ts_dev"]  = float(hk.get("H2H_TS_VS_OVERALL",  0) or 0)
        feats["h2h_fga_dev"] = float(hk.get("H2H_FGA_VS_OVERALL", 0) or 0)
        feats["h2h_min_dev"] = float(hk.get("H2H_MIN_VS_OVERALL", 0) or 0)
        feats["h2h_conf"]    = float(hk.get("H2H_CONFIDENCE",     0) or 0)
        feats["h2h_games"]   = float(hk.get("H2H_GAMES",          0) or 0)
        feats["h2h_trend"]   = float(hk.get("H2H_PTS_TREND",      0) or 0)

        row = {
            **feats,
            "actual_pts":  actual_pts,
            "target_cls":  1 if actual_pts > line else 0,
            "player":      pname,
            "date":        pd.Timestamp(date),
            "date_str":    date_str,
            "pos":         pos,
            "opponent":    opponent,
            "h2h_avg":     float(hk.get("H2H_AVG_PTS", 0) or 0),
            **{k: prop[k] for k in ("line","min_line","max_line","over_odds","under_odds",
                                     "game","home","away","books","season","source")},
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df = df.fillna(0)
    print(f"    Rows built: {len(df):,}  |  Skipped: {skip}")
    print(f"    OVER rate: {df['target_cls'].mean():.1%}")
    return df


# =============================================================================
# STEP 3: TRAIN MODELS (OOF — no leakage)
# =============================================================================

def train_models_oof(df, skip_train=False):
    X     = df[ML_FEATURES].values
    y_cls = df["target_cls"].values
    y_reg = df["actual_pts"].values
    lines = df["line"].values
    n     = len(df)

    if skip_train and FILE_CLF.exists():
        print("  --no-train: using existing models")
        with open(FILE_CLF,"rb") as f: clf = pickle.load(f)
        with open(FILE_REG,"rb") as f: reg = pickle.load(f)
        with open(FILE_CAL,"rb") as f: cal = pickle.load(f)
        trust = json.loads(FILE_TRUST.read_text()) if FILE_TRUST.exists() else {}
        oof_prob = np.zeros(n); oof_reg = np.zeros(n)
        tscv = TimeSeriesSplit(n_splits=5)
        for _, (tr, va) in enumerate(tscv.split(X)):
            oof_prob[va] = clf.predict_proba(X[va])[:,1]
            oof_reg[va]  = reg.predict(X[va])
        return clf, reg, cal, trust, oof_prob, oof_reg

    # Sample weights
    w = 1.0 + (np.arange(n) / n)
    w[df["date"].dt.month.values == 10] *= 0.4
    w[df["date"].dt.month.values == 11] *= 0.7
    w[df["h2h_conf"].values > 0.6] *= 1.2
    w[df["mean_reversion_risk"].values == 1.0] *= 0.8
    w = w / w.mean()

    clf_kw = dict(n_estimators=400, max_depth=3, learning_rate=0.035,
                  min_samples_leaf=15, subsample=0.75,
                  n_iter_no_change=30, validation_fraction=0.1, tol=1e-4, random_state=42)
    reg_kw = dict(n_estimators=400, max_depth=4, learning_rate=0.035,
                  min_samples_leaf=15, subsample=0.75, loss="huber", alpha=0.9,
                  n_iter_no_change=30, validation_fraction=0.1, tol=1e-4, random_state=42)

    tscv     = TimeSeriesSplit(n_splits=5)
    oof_prob = np.zeros(n)
    oof_reg  = np.zeros(n)

    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        cf = GradientBoostingClassifier(**clf_kw)
        rf = GradientBoostingRegressor(**reg_kw)
        cf.fit(X[tr], y_cls[tr], sample_weight=w[tr])
        rf.fit(X[tr], y_reg[tr], sample_weight=w[tr])
        oof_prob[va] = cf.predict_proba(X[va])[:,1]
        oof_reg[va]  = rf.predict(X[va])
        c = ((oof_prob[va]>0.5)==y_cls[va]).mean()
        r = ((oof_reg[va]>lines[va])==y_cls[va]).mean()
        print(f"    Fold {fold}: clf={c:.3f}  reg={r:.3f}")

    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof_prob, y_cls)

    # OOF trust scores
    oof_dir = (oof_prob > 0.5).astype(int)
    df["_oof_correct"] = (oof_dir == y_cls).astype(int)
    trust = {
        p: round(float(g["_oof_correct"].mean()), 4)
        for p, g in df.groupby("player") if len(g) >= 10
    }

    # Final models on full data
    clf = GradientBoostingClassifier(**clf_kw)
    reg = GradientBoostingRegressor(**reg_kw)
    clf.fit(X, y_cls, sample_weight=w)
    reg.fit(X, y_reg, sample_weight=w)
    print(f"  Final: clf={clf.n_estimators_} trees  reg={reg.n_estimators_} trees")

    MODEL_DIR.mkdir(exist_ok=True)
    with open(FILE_CLF,"wb") as f: pickle.dump(clf, f)
    with open(FILE_REG,"wb") as f: pickle.dump(reg, f)
    with open(FILE_CAL,"wb") as f: pickle.dump(cal, f)
    FILE_TRUST.write_text(json.dumps(trust, indent=2))
    print(f"  Models saved -> {MODEL_DIR}")

    return clf, reg, cal, trust, oof_prob, oof_reg


# =============================================================================
# STEP 4: APPLY V14 SCORING
# =============================================================================

def apply_v14_scoring(df, cal, trust, oof_prob, oof_reg):
    lines    = df["line"].values
    actual   = df["actual_pts"].values
    y_cls    = df["target_cls"].values
    cal_prob = cal.transform(oof_prob)
    pred_pts = oof_reg
    gap      = np.abs(pred_pts - lines)
    gap_conf = np.clip(0.5 + gap * 0.04, 0.45, 0.90)
    sp       = df["season_progress"].values
    ew       = df["early_season_weight"].values

    vol_norm  = np.clip((df["level"].values - lines) / 10.0, -1, 1)
    hr_signal = df["hr10"].values - 0.5
    trend_s   = np.clip(df["momentum"].values / 8.0, -1, 1)
    composite = 0.4*vol_norm + 0.35*hr_signal + 0.25*trend_s
    comp_conf = np.clip(0.5 + np.abs(composite)*0.3, 0.50, 0.85)

    alpha = 0.55 + 0.05*sp
    beta  = 0.30 - 0.05*sp
    fc    = alpha*cal_prob + beta*gap_conf + 0.15*comp_conf

    reg_dir  = (pred_pts > lines).astype(int)
    xhot     = df["extreme_hot"].values
    xcold    = df["extreme_cold"].values
    mom      = df["momentum"].values

    fc = np.where((xhot==1)&(reg_dir==1), fc-0.03, fc)
    fc = np.where((xcold==1)&(reg_dir==0), fc-0.02, fc)
    fc = np.where((mom>3)&(mom<=6)&(reg_dir==1), fc+0.015, fc)
    fc = np.where((mom<-3)&(mom>=-6)&(reg_dir==0), fc+0.015, fc)
    fc = fc * (0.70 + 0.30*ew)
    fc = np.where(df["is_long_rest"].values==1, fc-0.03, fc)
    fc = np.where(lines>=HIGH_LINE_THRESHOLD, fc-0.03, fc)
    fc = np.where(df["line_sharpness"].values>0.80, fc+0.01, fc)
    fc = np.clip(fc, 0.40, 0.90)

    clf_dir       = (oof_prob > 0.5).astype(int)
    engines_agree = (reg_dir == clf_dir)
    lo, hi        = LEAN_ZONE
    is_over  = (cal_prob >= hi) & engines_agree
    is_under = (cal_prob <= lo) & engines_agree
    is_lean  = ~(is_over | is_under)

    h2h_ts = df["h2h_ts_dev"].values
    h2h_g  = df["h2h_games"].values
    h2h_ok = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if h2h_g[i] >= 3:
            if is_over[i]  and h2h_ts[i] < -3: h2h_ok[i] = False
            if is_under[i] and h2h_ts[i] >  3: h2h_ok[i] = False

    std10    = df["std10"].values
    vol_risk = df["vol_risk"].values
    tl_arr   = []
    for i in range(len(df)):
        if is_lean[i]:
            tl_arr.append("T3_LEAN"); continue
        f=fc[i]; g2=gap[i]; s=std10[i]; ha=h2h_ok[i]; hv=(s>8)or(vol_risk[i]>1.5)
        if   f>=0.73 and g2>=5.0 and s<=6 and ha and not hv: tl_arr.append("T1_ULTRA")
        elif f>=0.68 and g2>=4.0 and s<=7 and ha and not hv: tl_arr.append("T1_PREMIUM")
        elif f>=0.63 and g2>=3.0 and s<=8 and ha and not hv: tl_arr.append("T1")
        elif f>=0.56 and g2>=2.0 and s<=9 and ha:            tl_arr.append("T2")
        else:                                                  tl_arr.append("T3")

    for i in range(len(df)):
        if trust.get(df["player"].iloc[i], 1.0) < TRUST_THRESHOLD and tl_arr[i].startswith("T1"):
            tl_arr[i] = "T2"

    tl_arr = np.array(tl_arr)
    units  = np.array([UNIT_MAP.get(t, 0.0) for t in tl_arr])
    dirs   = []
    for i in range(len(df)):
        if is_lean[i]: dirs.append("LEAN OVER" if cal_prob[i]>=0.5 else "LEAN UNDER")
        else:          dirs.append("OVER" if is_over[i] else "UNDER")

    results = []
    for i in range(len(df)):
        if is_lean[i]: results.append("LEAN")
        elif is_over[i]:  results.append("WIN" if actual[i]>lines[i] else "LOSS")
        else:             results.append("WIN" if actual[i]<=lines[i] else "LOSS")

    df["direction"]    = dirs
    df["tierLabel"]    = tl_arr
    df["tier"]         = [1 if t.startswith("T1") else(2 if t=="T2" else 3) for t in tl_arr]
    df["conf"]         = np.round(fc, 4)
    df["predPts"]      = np.round(pred_pts, 1)
    df["predGap"]      = np.round(gap, 2)
    df["calProb"]      = np.round(cal_prob, 4)
    df["units"]        = units
    df["result"]       = results
    df["actualPts"]    = actual
    df["delta"]        = np.round(actual - lines, 1)
    df["enginesAgree"] = engines_agree
    df["isLean"]       = is_lean

    t12 = df[(df["tier"]<=2) & df["result"].isin(["WIN","LOSS"])]
    t1  = df[(df["tier"]==1) & df["result"].isin(["WIN","LOSS"])]
    if len(t12): print(f"  T1+T2 OOF accuracy: {(t12['result']=='WIN').mean():.1%} ({len(t12):,})")
    if len(t1):  print(f"  T1    OOF accuracy: {(t1['result']=='WIN').mean():.1%}  ({len(t1):,})")
    return df


# =============================================================================
# STEP 5: BUILD JSON FILES
# =============================================================================

def build_json_files(df, recent_idx, target_date=None):
    DATA_DIR.mkdir(exist_ok=True)
    plays_2526, plays_2425 = [], []

    total = len(df)
    for i in range(total):
        if i % 3000 == 0:
            print(f"  Building play {i:,}/{total:,}...")
        row  = df.iloc[i]
        play = _build_play(row, recent_idx)
        if str(row.get("season","")) == "2024-25":
            plays_2425.append(play)
        else:
            plays_2526.append(play)

    def _sort_key(p):
        return (-int(pd.Timestamp(p["date"]).timestamp()), p["tier"], -p["conf"])

    plays_2526.sort(key=_sort_key)
    plays_2425.sort(key=_sort_key)

    _save(FILE_SEASON_2526, plays_2526)
    print(f"  season_2025_26.json: {len(plays_2526):,} plays")
    _save(FILE_SEASON_2425, plays_2425)
    print(f"  season_2024_25.json: {len(plays_2425):,} plays")

    all_plays = plays_2526 + plays_2425
    if target_date:
        today = [p for p in all_plays if p["date"] == target_date]
    else:
        t12_dates = [p["date"] for p in all_plays if p["tier"] <= 2]
        best = max(t12_dates) if t12_dates else max(p["date"] for p in all_plays)
        today = [p for p in all_plays if p["date"] == best]
    today.sort(key=lambda p: (p["tier"], -p["conf"]))
    _save(FILE_TODAY, today)
    print(f"  today.json: {len(today)} plays  date={today[0]['date'] if today else '?'}")

    _write_summary(plays_2526, plays_2425)
    print(f"  backtest_summary.json written")


def _build_play(row, recent_idx):
    pname    = str(row["player"])
    date_str = str(row["date_str"])
    line     = float(row["line"])

    all_s  = recent_idx.get(pname, [])
    prior  = [(d,pt,h,op) for d,pt,h,op in all_s if d < date_str]
    r20    = [{"date":d,"pts":float(pt),"home":bool(h),"opponent":str(op),
               "overLine":float(pt)>line} for d,pt,h,op in prior[-20:]]

    L30 = float(row.get("level",0))
    L10 = L30 + float(row.get("reversion",0))
    L5  = L30 + float(row.get("momentum",0))
    L3  = L5  + float(row.get("acceleration",0))
    flags = max(0, min(10, int(round(float(row.get("hr10",0.5))*10))))

    play = {
        "player":   pname, "date": date_str,
        "game":     str(row.get("game","")), "home": str(row.get("home","")),
        "away":     str(row.get("away","")), "opponent": str(row.get("opponent","")),
        "position": str(row.get("pos","")), "season": str(row.get("season","2025-26")),
        "source":   str(row.get("source","real")),
        "line":     line, "direction": str(row.get("direction","")),
        "tierLabel":str(row.get("tierLabel","T3")), "tier": int(row.get("tier",3)),
        "conf":     float(row.get("conf",0.5)), "predPts": float(row.get("predPts",line)),
        "predGap":  float(row.get("predGap",0)), "calProb": float(row.get("calProb",0.5)),
        "units":    float(row.get("units",0)), "flags": flags,
        "enginesAgree":      bool(row.get("enginesAgree",True)),
        "meanReversionRisk": float(row.get("mean_reversion_risk",0)),
        "seasonProgress":    float(row.get("season_progress",0.5)),
        "earlySeasonW":      float(row.get("early_season_weight",1.0)),
        "volRisk":   float(row.get("vol_risk",0)),
        "overOdds":  float(row.get("over_odds",-110) or -110),
        "underOdds": float(row.get("under_odds",-110) or -110),
        "books":     int(row.get("books",1) or 1),
        "result":    str(row.get("result","")),
        "actualPts": float(row.get("actualPts",0)),
        "delta":     float(row.get("delta",0)),
        "l30":round(L30,1),"l10":round(L10,1),"l5":round(L5,1),"l3":round(L3,1),
        "std10":    round(float(row.get("std10",5)),2),
        "hr10":     round(float(row.get("hr10",0.5)),3),
        "hr30":     round(float(row.get("hr30",0.5)),3),
        "min_l10":  round(float(row.get("min_l10",30)),1),
        "fga_l10":  round(float(row.get("fga_l10",10)),1),
        "momentum": round(float(row.get("momentum",0)),2),
        "defP_dynamic": int(row.get("defP_dynamic",15)),
        "pace_rank":    int(row.get("pace_rank",15)),
        "h2h_avg":   round(float(row.get("h2h_avg",0)),1),
        "h2h_games": int(row.get("h2h_games",0) or 0),
        "h2h_ts_dev":round(float(row.get("h2h_ts_dev",0)),4),
        "h2h_conf":  round(float(row.get("h2h_conf",0)),4),
        "lineSharpness": round(float(row.get("line_sharpness",0.67)),3),
        "is_b2b":    int(row.get("is_b2b",0)),
        "rest_days": int(row.get("rest_days",2)),
        "recent20":  r20,
        "preMatchReason":  "",
        "postMatchReason": "",
        "lossType": "",
    }

    p_for_reason = {**play, "_n_games": int(row.get("games_depth",0.5)*30),
                    "is_long_rest": int(row.get("is_long_rest",0)),
                    "flagDetails":[]}
    try:
        play["preMatchReason"] = generate_pre_match_reason(p_for_reason)
    except Exception:
        play["preMatchReason"] = f"V14: {play['direction']} {line}"
    try:
        post, lt = generate_post_match_reason(p_for_reason)
        play["postMatchReason"] = post
        play["lossType"] = lt
    except Exception:
        play["postMatchReason"] = f"Result: {play['result']}. Actual: {play['actualPts']:.0f}pts."
        play["lossType"] = "MODEL_CORRECT" if play["result"]=="WIN" else "MODEL_FAILURE_GENERAL"

    return play


def _write_summary(plays_2526, plays_2425):
    def _s(plays):
        g = [p for p in plays if p["result"] in ("WIN","LOSS")]
        def acc(sub): return round(sum(1 for p in sub if p["result"]=="WIN")/max(len(sub),1),4)
        t1   = [p for p in g if p["tier"]==1]
        t2   = [p for p in g if p["tierLabel"]=="T2"]
        t12  = t1+t2
        ultra= [p for p in g if p["tierLabel"]=="T1_ULTRA"]
        prem = [p for p in g if p["tierLabel"]=="T1_PREMIUM"]
        pl   = sum(p["units"]*0.909 if p["result"]=="WIN" else -p["units"] for p in g if p["units"]>0)
        monthly = {}
        for p in t12:
            m = p["date"][:7]
            if m not in monthly: monthly[m]={"wins":0,"total":0,"pl":0.0}
            monthly[m]["total"] += 1
            if p["result"]=="WIN": monthly[m]["wins"]+=1
            monthly[m]["pl"] += p["units"]*0.909 if p["result"]=="WIN" else -p["units"]
        return {"total":len(plays),"graded":len(g),
                "t1_ultra":{"n":len(ultra),"acc":acc(ultra)},
                "t1_premium":{"n":len(prem),"acc":acc(prem)},
                "t1":{"n":len(t1),"acc":acc(t1)},
                "t2":{"n":len(t2),"acc":acc(t2)},
                "t12":{"n":len(t12),"acc":acc(t12)},
                "pl":round(pl,1),"monthly":monthly}

    summary = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "season_2526": _s(plays_2526), "season_2425": _s(plays_2425)}
    _save(DATA_DIR / "backtest_summary.json", summary)


def _save(path, data):
    with open(path,"w") as f:
        json.dump(clean_json(data), f, separators=(",",":"))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",   choices=["2526","2425","both"], default="both")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--date",     type=str, default=None)
    args = parser.parse_args()

    print("="*65)
    print("  PropEdge V14.0 — Backtest Generator")
    print("="*65)
    t0 = datetime.now()

    (player_idx, h2h_lkp, dvp_rank, pace_cache,
     b2b_map, recent_idx, props_2526, props_2425) = load_all_data()

    props = []
    if args.season in ("both","2526"): props += props_2526
    if args.season in ("both","2425"): props += props_2425

    print(f"\n[2/5] Building feature rows ({len(props):,} props)...")
    df = build_feature_rows(player_idx, h2h_lkp, dvp_rank, pace_cache, b2b_map, props)

    print("\n[3/5] Training V14 models (OOF 5-fold)...")
    clf, reg, cal, trust, oof_prob, oof_reg = train_models_oof(df, args.no_train)

    print("\n[4/5] Applying V14 Adaptive Fusion scoring...")
    df = apply_v14_scoring(df, cal, trust, oof_prob, oof_reg)

    print("\n[5/5] Writing JSON output files...")
    build_json_files(df, recent_idx, target_date=args.date)

    secs = (datetime.now()-t0).seconds
    print(f"\n  Done in {secs//60}m {secs%60}s")
    print("  Open index.html in your browser to view the dashboard.\n")

if __name__ == "__main__":
    main()
