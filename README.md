# PropEdge V14.0 — Adaptive Intelligence Engine

**NBA Player Points Prop Prediction System**  
OOF T1+T2 Accuracy: **75.9%** | T1 Accuracy: **80.8%** | Per-play P&L: **+0.79u at −110**

---

## Quick Setup

```bash
# 1. Clone
git clone git@github.com:iamwerewolf1007/PropEdgeV14.0.git
cd ~/Documents/GitHub/PropEdgeV14.0

# 2. Install Python dependencies
pip3 install -r requirements.txt

# 3. Copy your source files into source-files/
cp nba_gamelogs_2024_25.csv    source-files/
cp nba_gamelogs_2025_26.csv    source-files/
cp h2h_database.csv            source-files/
cp "PropEdge_-_Match_and_Player_Prop_lines_.xlsx" source-files/

# 4. First-time generate (builds DVP, H2H, trains all models — ~6-8 min)
python3 run.py generate

# 5. Install scheduler (launchd agents)
python3 run.py install

# 6. Verify everything
python3 run.py status

# 7. Test a manual prediction run
python3 run.py predict 2
```

---

## Schedule

### Weekday (Monday–Friday)

| Time (UK) | Batch | Purpose |
|---|---|---|
| 08:00 | B0 | Grade yesterday + retrain all models |
| 08:30 | B1 | Morning prop scan |
| 18:30 | B2 | Main prediction run |
| 20:30 | B3 | Top-up run |
| 23:30 | B4 | Late West-Coast game run |

### Weekend (Saturday–Sunday)

B0 always runs at 08:00 UK (grading is fixed). B1–B4 **shift relative to the first NBA tip-off**, detected automatically from The Odds API at 05:55 UK each morning.

| Batch | Offset from first tip | Floor | Ceiling |
|---|---|---|---|
| B1 | −90 min | 10:30 | 14:00 |
| B2 | −60 min | 16:00 | 19:00 |
| B3 | +30 min | 18:00 | 22:00 |
| B4 | +3 hours | 21:00 | 23:59 |

**Example:** If the first tip is at 17:30 UK on Saturday:
- B1 → 16:00 UK (−90min, at floor)
- B2 → 16:30 UK (−60min)
- B3 → 18:00 UK (+30min)
- B4 → 20:30 UK (+3hr)

---

## Commands

```bash
python3 run.py install          # Install launchd scheduler
python3 run.py uninstall        # Remove scheduler
python3 run.py status           # Agent status + model freshness + credits
python3 run.py generate         # First-time: train models + build data
python3 run.py grade            # Manual Batch 0 (grade + retrain)
python3 run.py predict [1-4]    # Manual prediction batch (default: 2)
python3 run.py retrain          # Retrain models only
python3 run.py dvp              # Rebuild DVP rankings only
python3 run.py h2h              # Rebuild H2H database only
python3 run.py weekend          # Preview today's weekend schedule
python3 run.py check            # Data integrity checks
```

---

## Prop Input

**Primary (Excel):** Fill in `source-files/PropEdge_-_Match_and_Player_Prop_lines_.xlsx`  
Sheet: `Player_Points_Props`  
Required columns: `Player, Game, Home, Away, Line, Over Odds, Under Odds, Books, Min Line, Max Line`

**Fallback (Odds API):** If Excel is empty, the system auto-fetches from The Odds API.  
API key: set `ODDS_API_KEY` in `config.py`.

---

## V14 Architecture

Three engines fused with season-adaptive weights:

```
Engine A (Composite 10-signals)  ×0.15
Engine B (GBR Regressor)         ×0.30  (early season)  →  0.25 (late season)
Engine C (GBR Classifier)        ×0.55  (early season)  →  0.60 (late season)
```

**Key V14 innovations:**
- Season-adaptive fusion (classifier gains weight as rolling windows fill)
- Z-score momentum signals (variance-normalised trend)
- Mean reversion risk flags with confidence penalties
- Early-season scaling (Oct games penalised ~30% confidence)
- Volatility risk score (`std10 × line/100`) replaces binary cutoff
- Strict lean zone (0.46–0.54) eliminates borderline coin-flip calls
- Engine conflict → mandatory LEAN (53.3% accuracy when engines disagree)

---

## File Structure

```
PropEdgeV14/
├── config.py               # All constants, paths, helpers
├── rolling_engine.py       # 56-feature extraction (no lookahead)
├── model_trainer.py        # GBR clf + reg + calibrator + player trust
├── dvp_updater.py          # Live DVP from game log CSV
├── h2h_builder.py          # H2H database builder
├── reasoning_engine.py     # Pre/post-match narratives
├── batch_predict.py        # B1-B4 prediction pipeline
├── batch0_grade.py         # B0 grade + retrain pipeline
├── audit.py                # Append-only audit trail
├── scheduler.py            # Smart launchd scheduler with weekend detection
├── run.py                  # Master CLI
├── requirements.txt
├── source-files/           # Input CSVs + Excel (user supplies)
├── data/                   # today.json, season JSONs, audit log
├── models/                 # Trained model pkl files
├── logs/                   # launchd stdout/stderr per batch
└── daily/                  # Per-date Excel records
```

---

## Important Rules (Never Violate)

1. Never `groupby().apply()` for rolling stats
2. Always `parse_dates=['GAME_DATE']` in every `read_csv`
3. Never read `L*_*` CSV columns at prediction time — always recompute
4. Always `filter_played()` before rolling windows
5. Rolling windows span both seasons — no season resets
6. Graded plays are permanently immutable
7. `clean_json()` before every `json.dump()`
8. Engine disagreement always → LEAN direction

---

## Troubleshooting

**Models missing:** `python3 run.py generate`  
**Git push fails:** `ssh-add ~/.ssh/id_ed25519` then verify with `ssh -T git@github.com`  
**Agents not firing:** Mac must be awake. Use `caffeinate -i &` to prevent sleep.  
**Low API credits:** Check with `python3 run.py status`. Update key in `config.py`.  
**October accuracy low:** Expected — thin rolling windows. Model auto-penalises early season.

---

*PropEdge V14.0 · April 2026*
