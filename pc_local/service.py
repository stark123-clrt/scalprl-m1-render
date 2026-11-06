"""SERVICE DU COMITÉ RL v4 (scalping or M5) — tourne sur le PC à côté de MetaTrader 5.

Échange par fichiers avec l'EA ScalpRL_Committee.mq5, dans le dossier commun de MT5
(%APPDATA%\\MetaQuotes\\Terminal\\Common\\Files\\scalp_rl) :
  - history.csv  : écrit par l'EA au démarrage, historique M1 (heure UTC, OHLC bid, spread $, volume) ;
  - request.csv  : écrit par l'EA à chaque clôture M5 : état de la position + 30 dernières bougies M1 ;
  - response.csv : écrit ici : action du comité (0 = à plat, 1 = BUY, 2 = SELL) + votes des 3 agents.
Les calculs sont ceux de l'entraînement (parité vérifiée sur Kaggle : notebook xau-rl-live-export).
Lancement : python service.py   (ou start_service.bat)
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # 1 cœur pour les calculs : bien moins de mémoire réservée
    os.environ.setdefault(_v, "1")
import gc, json, sys, time, traceback, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features
from phase1 import ohlc, ind, TF
from strategies import strategy_signals
features.ohlc, features.ind, features.TF, features.strategy_signals = ohlc, ind, TF, strategy_signals

BRIDGE = Path(os.environ["APPDATA"]) / "MetaQuotes" / "Terminal" / "Common" / "Files" / "scalp_rl"
LOGS = HERE / "logs"
MAX_HOLD, TARGET = 12, 20                       # mêmes constantes que l'entraînement (60 min, objectif doux 20 trades/j)
WINDOW = 400_000                                # minutes M1 gardées (parité vérifiée avec l'historique complet)
DUKAS_SPREAD_MED = 0.68                         # spread médian Dukascopy 2026 ($) vu à l'entraînement
GOLDSTARK_URL = "https://goldstark.onrender.com/signal"
USE_GOLDSTARK = False                           # retiré le 22/10/2026 : ses probabilités en direct (0,75-0,88) poussaient
                                                # le comité à acheter pendant les baisses -> avis neutre pour l'agent

def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line, flush=True)
    with open(LOGS / "service.log", "a", encoding="utf-8") as f: f.write(line + "\n")

def load_models():
    M = HERE / "models"
    norm = np.load(M / "norm_v4.npz", allow_pickle=True)
    boosters = {h: lgb.Booster(model_file=str(M / f"lgb_h{h}.txt")) for h in features.ADV_H}
    policies = [dict(np.load(M / f"policy_seed{s}.npz")) for s in (6, 3, 5)]
    return norm["mu"], norm["sd"], list(norm["cols"]), boosters, policies

def read_bars(lines):
    """Lignes 'temps_utc,o,h,l,c,spread,volume' (prix BID) -> DataFrame M1 au prix mid."""
    a = np.array([[float(x) for x in l.split(",")] for l in lines if l.strip()])
    if len(a) == 0: return pd.DataFrame(columns=["o", "h", "l", "c", "spread", "v"])
    hs = a[:, 5] / 2
    df = pd.DataFrame({"o": a[:, 1] + hs, "h": a[:, 2] + hs, "l": a[:, 3] + hs, "c": a[:, 4] + hs, "spread": a[:, 5], "v": a[:, 6]},
                      index=pd.to_datetime(a[:, 0].astype(np.int64), unit="s"))
    df.index.name = "datetime"
    return df[~df.index.duplicated(keep="last")].sort_index()

_gs_cache = {"t": None, "p": np.nan}
def goldstark_prob():
    """Probabilité de hausse GoldStark (6e avis). En cas d'indisponibilité : NaN (= valeur neutre pour l'agent)."""
    now = time.time()
    if _gs_cache["t"] and now - _gs_cache["t"] < 120: return _gs_cache["p"]
    try:
        with urllib.request.urlopen(GOLDSTARK_URL, timeout=8) as r: d = json.loads(r.read().decode())
        p = d.get("probability"); _gs_cache["p"] = float(p) if p is not None else np.nan
    except Exception as e:
        _gs_cache["p"] = np.nan; log(f"GoldStark indisponible ({type(e).__name__}) -> avis neutre")
    _gs_cache["t"] = now
    return _gs_cache["p"]

def decide(m1, t_dec, state, models):
    mu, sd, cols, boosters, policies = models
    w = m1[m1.index < t_dec].iloc[-WINDOW:].copy()
    med = w.spread.iloc[-20_000:].median()                                   # spread du courtier -> échelle Dukascopy
    if med > 0: w["spread"] = w.spread * (DUKAS_SPREAD_MED / med)
    Fp = features.advisor_features(w)
    last = Fp.index[-1]
    if last + pd.Timedelta(minutes=5) != t_dec:
        raise ValueError(f"dernière bougie M5 {last} incohérente avec la décision {t_dec} (bougies M1 manquantes ?)")
    adv = pd.DataFrame({f"adv_{h}": boosters[h].predict(Fp.iloc[[-1]].values) for h in features.ADV_H}, index=[last])
    gs = pd.Series([goldstark_prob() if USE_GOLDSTARK else np.nan], index=[last])
    F, close, atr5 = features.agent_features(w, adv, gs)
    x = np.clip(np.nan_to_num((F[cols].iloc[-1].values.astype(np.float64) - mu) / sd), -10, 10)
    a = max(float(atr5.iloc[-1]), 1e-6)
    st = np.array([state["pos"], np.clip(state["upnl"] / a, -10, 10), state["age"] / MAX_HOLD, min(state["n_day"] / TARGET, 3)])
    obs = np.concatenate([x, st]).astype(np.float32).astype(np.float64)
    action, votes = features.committee_action(policies, obs)
    why = features.explain(policies, x, st, features.feature_groups(cols), action)
    return action, votes, dict(why=why, adv={k: round(float(v), 4) for k, v in adv.iloc[0].items()}, gs=gs.iloc[0], atr=a,
                               close=float(close.iloc[-1]))

def write_atomic(path, text):
    tmp = path.with_suffix(".tmp"); tmp.write_text(text, encoding="ascii"); os.replace(tmp, path)

def main():
    BRIDGE.mkdir(parents=True, exist_ok=True); LOGS.mkdir(exist_ok=True)
    models = load_models()
    log(f"Comité v4 chargé (agents 6, 3, 5 ; {len(models[2])} features). Dossier d'échange : {BRIDGE}")
    m1, hist_mtime = None, None
    hist, req, resp = BRIDGE / "history.csv", BRIDGE / "request.csv", BRIDGE / "response.csv"
    while True:
        try:
            if hist.exists() and hist.stat().st_mtime != hist_mtime and not (BRIDGE / "history.tmp").exists():
                hist_mtime = hist.stat().st_mtime
                m1 = read_bars(hist.read_text(encoding="ascii").splitlines()[1:])
                days = (m1.index[-1] - m1.index[0]).days if len(m1) else 0
                log(f"Historique reçu : {len(m1):,} minutes M1 ({days} jours), dernière {m1.index[-1] if len(m1) else '-'}")
                gc.collect()                                     # rend la mémoire des calculs à Windows
                if len(m1) and (pd.Timestamp.now("UTC").tz_localize(None) - m1.index[-1]).days > 3:
                    log(f"ATTENTION : historique ancien (dernière minute {m1.index[-1]}) -> attente d'un nouvel envoi par l'EA"); m1 = None; continue
                if days < 300: log("ATTENTION : moins de 300 jours d'historique -> indicateurs D1 incomplets "
                                   "(MT5 : Outils > Options > Graphiques > Barres max = Illimité, puis redémarrer l'EA)")
            if req.exists():
                lines = req.read_text(encoding="ascii").splitlines(); req.unlink()
                rid, t_dec, pos, upnl, age, n_day = lines[0].split(",")
                if m1 is None: log("Requête reçue mais pas encore d'historique"); continue
                t0 = time.time()
                new = read_bars(lines[1:])
                if len(new) and new.index[0] - m1.index[-1] > pd.Timedelta(minutes=10):
                    (BRIDGE / "need_history.flag").write_text("1", encoding="ascii")    # trou : l'EA renvoie tout l'historique
                    log(f"Historique incomplet (trou de {m1.index[-1]} à {new.index[0]}) -> l'EA renvoie l'historique, pas de décision")
                    m1 = None; continue
                m1 = pd.concat([m1[~m1.index.isin(new.index)], new]).sort_index().iloc[-WINDOW - 5000:]
                t_dec = pd.to_datetime(int(t_dec), unit="s")
                state = dict(pos=int(pos), upnl=float(upnl), age=int(age), n_day=int(n_day))
                action, votes, info = decide(m1, t_dec, state, models)
                write_atomic(resp, f"{rid},{action},{votes[0]},{votes[1]},{votes[2]}\n")
                gc.collect()                                     # rend la mémoire des calculs à Windows
                name = ("A PLAT", "BUY", "SELL")[action]
                log(f"{t_dec:%Y-%m-%d %H:%M} UTC | {name:6s} | votes {votes} | pos {state['pos']:+d} age {state['age']} "
                    f"trades/j {state['n_day']} | portée par : {info['why']} | close {info['close']:.2f} ATR {info['atr']:.2f} | GoldStark {info['gs'] if USE_GOLDSTARK else 'retiré'} "
                    f"| conseillers {info['adv']} | {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"ERREUR : {e}\n{traceback.format_exc()}")
        time.sleep(0.3)

if __name__ == "__main__":
    main()
