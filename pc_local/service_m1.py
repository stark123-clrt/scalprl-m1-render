"""SERVICE DU COMITÉ OR M1 (suivi de tendance EMA50 M15) — même principe que service_v5.py, dossier d'échange séparé.
EA : ScalpRL_Trend_M1 sur un graphique XAUUSD (réglages par défaut : M1, scalp_rl_m1, magique 260501, 60 min max).
Agents du comité : models_m1/committee.json (agents 1, 2, 3, 5, 6). Lancement : start_service_m1.bat
Décision à la clôture de CHAQUE minute : la réponse doit partir en quelques secondes (temps affiché dans le journal).
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # 1 cœur pour les calculs : bien moins de mémoire réservée
    os.environ.setdefault(_v, "1")
import gc, json, sys, time, traceback
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features_m1, features_v5
from features import committee_action, explain
from service import read_bars, write_atomic

BRIDGE = Path(os.environ["APPDATA"]) / "MetaQuotes" / "Terminal" / "Common" / "Files" / "scalp_rl_m1"
LOGS = HERE / "logs"
MAX_HOLD, TARGET, WINDOW, DUKAS_SPREAD_MED = 60, 20, 150_000, 0.62   # fenêtre : exacte dès 100 000 min (kernel xau-m1-live-export) ; spread médian Dukascopy 2025-2026 ($)
TREND_COLS = ["m15_d_ema50", "m15_d_ema50"]

def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line, flush=True)
    with open(LOGS / "service_m1.log", "a", encoding="utf-8") as f: f.write(line + "\n")

def load_models():
    M = HERE / "models_m1"
    norm = np.load(M / "norm_m1.npz", allow_pickle=True)
    cfg = json.loads((M / "committee.json").read_text(encoding="utf-8")); seeds = cfg["seeds"]
    global TREND_COLS, MAX_HOLD, TARGET
    TREND_COLS = cfg.get("trend_rule", TREND_COLS)
    MAX_HOLD, TARGET = cfg.get("max_hold", MAX_HOLD), cfg.get("target", TARGET)
    policies = [dict(np.load(M / f"policy_m1_seed{s}.npz")) for s in seeds]
    return norm["mu"], norm["sd"], list(norm["cols"]), policies, seeds

def decide(m1, t_dec, state, models):
    mu, sd, cols, policies, _ = models
    w = m1[m1.index < t_dec].iloc[-WINDOW:].copy()
    med = w.spread.iloc[-20_000:].median()
    if med > 0: w["spread"] = w.spread * (DUKAS_SPREAD_MED / med)
    F, close, atr = features_m1.agent_features_m1(w)
    last = F.index[-1]
    if last + pd.Timedelta(minutes=1) != t_dec:
        raise ValueError(f"dernière bougie M1 {last} incohérente avec la décision {t_dec}")
    row = F[cols].iloc[-1]
    x = np.clip(np.nan_to_num((row.values.astype(np.float64) - mu) / sd), -10, 10)
    a = max(float(atr.iloc[-1]), 1e-6)
    st = np.array([state["pos"], np.clip(state["upnl"] / a, -10, 10), state["age"] / MAX_HOLD, min(state["n_day"] / TARGET, 3)])
    raw, votes = committee_action(policies, np.concatenate([x, st]).astype(np.float32).astype(np.float64))
    allowed = features_v5.allowed_direction(row[TREND_COLS[0]], row[TREND_COLS[1]])
    action = features_v5.strict_rule(raw, state["pos"], allowed)
    why = explain(policies, x, st, features_m1.feature_groups_m1(cols), raw)
    return action, votes, dict(raw=raw, allowed=allowed, why=why, atr=a, close=float(close.iloc[-1]))

def main():
    BRIDGE.mkdir(parents=True, exist_ok=True); LOGS.mkdir(exist_ok=True)
    models = load_models()
    log(f"Comité OR M1 chargé (agents {models[4]} ; {len(models[2])} features ; règle de tendance {TREND_COLS} ; "
        f"fenêtre {WINDOW:,} min). Dossier d'échange : {BRIDGE}")
    m1, hist_mtime = None, None
    hist, req, resp = BRIDGE / "history.csv", BRIDGE / "request.csv", BRIDGE / "response.csv"
    names = ("A PLAT", "BUY", "SELL")
    while True:
        try:
            if hist.exists() and hist.stat().st_mtime != hist_mtime and not (BRIDGE / "history.tmp").exists():
                hist_mtime = hist.stat().st_mtime
                m1 = read_bars(hist.read_text(encoding="ascii").splitlines()[1:]).iloc[-WINDOW - 5000:]
                log(f"Historique reçu : {len(m1):,} minutes M1 gardées, dernière {m1.index[-1] if len(m1) else '-'}")
                gc.collect()                                     # rend la mémoire des calculs à Windows
            if req.exists():
                lines = req.read_text(encoding="ascii").splitlines(); req.unlink()
                rid, t_dec, pos, upnl, age, n_day = lines[0].split(",")
                if m1 is None: log("Requête reçue mais pas encore d'historique"); continue
                t0 = time.time()
                new = read_bars(lines[1:])
                if len(new) and new.index[0] - m1.index[-1] > pd.Timedelta(minutes=10):
                    (BRIDGE / "need_history.flag").write_text("1", encoding="ascii")
                    log(f"Historique incomplet (trou de {m1.index[-1]} à {new.index[0]}) -> l'EA renvoie l'historique, pas de décision")
                    m1 = None; continue
                m1 = pd.concat([m1[~m1.index.isin(new.index)], new]).sort_index().iloc[-WINDOW - 5000:]
                t_dec = pd.to_datetime(int(t_dec), unit="s")
                state = dict(pos=int(pos), upnl=float(upnl), age=int(age), n_day=int(n_day))
                action, votes, info = decide(m1, t_dec, state, models)
                v = (votes + [0, 0, 0])[:3]                       # l'EA lit 3 votes (affichage seulement)
                write_atomic(resp, f"{rid},{action},{v[0]},{v[1]},{v[2]}\n")
                gc.collect()                                     # rend la mémoire des calculs à Windows
                trend = {1.0: "HAUSSE", -1.0: "BAISSE", 0.0: "aucune"}[info["allowed"]]
                log(f"{t_dec:%Y-%m-%d %H:%M} UTC | {names[action]:6s} | comité {names[info['raw']]} votes {votes} | tendance {trend} "
                    f"| pos {state['pos']:+d} age {state['age']} | portée par : {info['why']} | close {info['close']:.2f} | {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"ERREUR : {e}\n{traceback.format_exc()}")
        time.sleep(0.3)

if __name__ == "__main__":
    main()
