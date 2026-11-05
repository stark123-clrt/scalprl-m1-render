"""SERVICE RENDER DU COMITÉ RL OR M1 (suivi de tendance EMA50 M15) — même cerveau que scalp_rl_live/service_m1.py,
mais l'échange se fait par Internet au lieu des fichiers du PC.

L'EA MT5 (mql5/ScalpRL_M1_Render.mq5) appelle ce service avec WebRequest :
  POST /history?part=K&last=0|1   historique M1 (lignes « temps_utc,o,h,l,c,spread,volume », prix BID), envoyé en morceaux
  POST /decide                    1re ligne « id,t_dec,pos,upnl,age,n_day » + les 30 dernières minutes M1
                                  -> « id,action,votes »  (action 0 = à plat, 1 = BUY, 2 = SELL)
                                  -> « need_history » si le service n'a pas (ou plus) l'historique : l'EA le renvoie
                                  -> « error,... » : pas de décision sur cette minute
  GET  /health                    état du service (pour Render) ; GET /log : dernières décisions (texte)
Les bougies viennent de MT5 (prix, spread et volume du compte Deriv) : décisions identiques au robot du PC.
Sécurité : si la variable d'environnement SCALPRL_TOKEN existe, chaque requête doit avoir l'en-tête X-Token identique.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import gc, json, threading, time, traceback
from collections import deque
from pathlib import Path
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

import features_m1, features_v5
from features import committee_action, explain

HERE = Path(__file__).resolve().parent
WINDOW = int(os.environ.get("SCALPRL_WINDOW", 110_000))      # minutes M1 de calcul (décisions exactes dès 100 000)
DUKAS_SPREAD_MED = 0.62                                       # spread médian Dukascopy 2025-2026 ($), comme le service du PC
TOKEN = os.environ.get("SCALPRL_TOKEN", "")
MAX_HOLD, TARGET, TREND_COLS = 60, 20, ["m15_d_ema50", "m15_d_ema50"]
NAMES = ("A PLAT", "BUY", "SELL")

LOG = deque(maxlen=600)
LOCK = threading.Lock()
S = {"m1": None, "parts": [], "last": None, "n_dec": 0, "start": time.time(), "secs": deque(maxlen=60)}

def log(msg):
    line = f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M:%S} UTC | {msg}"
    print(line, flush=True); LOG.append(line)

def read_bars(lines):
    """Lignes 'temps_utc,o,h,l,c,spread,volume' (prix BID) -> DataFrame M1 au prix mid (copie de service.py)."""
    a = np.array([[float(x) for x in l.split(",")] for l in lines if l.strip()])
    if len(a) == 0: return pd.DataFrame(columns=["o", "h", "l", "c", "spread", "v"])
    hs = a[:, 5] / 2
    df = pd.DataFrame({"o": a[:, 1] + hs, "h": a[:, 2] + hs, "l": a[:, 3] + hs, "c": a[:, 4] + hs, "spread": a[:, 5], "v": a[:, 6]},
                      index=pd.to_datetime(a[:, 0].astype(np.int64), unit="s"))
    df.index.name = "datetime"
    return df[~df.index.duplicated(keep="last")].sort_index()

def load_models():
    global TREND_COLS, MAX_HOLD, TARGET
    M = HERE / "models_m1"
    norm = np.load(M / "norm_m1.npz", allow_pickle=True)
    cfg = json.loads((M / "committee.json").read_text(encoding="utf-8"))
    TREND_COLS = cfg.get("trend_rule", TREND_COLS)
    MAX_HOLD, TARGET = cfg.get("max_hold", MAX_HOLD), cfg.get("target", TARGET)
    policies = [dict(np.load(M / f"policy_m1_seed{s}.npz")) for s in cfg["seeds"]]
    return norm["mu"], norm["sd"], list(norm["cols"]), policies, cfg["seeds"]

MODELS = load_models()

def decide(m1, t_dec, state):
    """Copie de service_m1.decide : fenêtre WINDOW, spread remis à l'échelle Dukascopy, vote du comité, règle stricte."""
    mu, sd, cols, policies, _ = MODELS
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
    return action, votes, dict(raw=raw, allowed=allowed, why=why, close=float(close.iloc[-1]))

app = FastAPI(title="Comité RL OR M1", docs_url=None, redoc_url=None)

def check(request: Request):
    if TOKEN and request.headers.get("x-token", "") != TOKEN:
        raise HTTPException(status_code=401, detail="token invalide")

@app.get("/")
@app.get("/health")
def health():
    m1 = S["m1"]
    return {"service": "comité RL OR M1", "agents": MODELS[4], "features": len(MODELS[2]), "fenetre": WINDOW,
            "historique_minutes": 0 if m1 is None else len(m1), "derniere_minute": None if m1 is None else str(m1.index[-1]),
            "decisions": S["n_dec"], "derniere_decision": S["last"],
            "calcul_moyen_s": round(float(np.mean(S["secs"])), 2) if S["secs"] else None,
            "en_ligne_depuis_min": round((time.time() - S["start"]) / 60, 1), "token": bool(TOKEN)}

@app.get("/log", response_class=PlainTextResponse)
def get_log(request: Request, n: int = 200):
    check(request)
    return "\n".join(list(LOG)[-n:])

@app.post("/history", response_class=PlainTextResponse)
async def history(request: Request, part: int = 0, last: int = 1):
    check(request)
    body = (await request.body()).decode("ascii", errors="ignore")
    lines = [l for l in body.splitlines() if l.strip() and not l.startswith("time_utc")]
    with LOCK:
        if part == 0: S["parts"] = []
        S["parts"].extend(lines)
        if not last:
            return f"ok,{len(S['parts'])}"
        try:
            m1 = read_bars(S["parts"]).iloc[-WINDOW - 5000:]
        finally:
            S["parts"] = []
        S["m1"] = m1; gc.collect()
        log(f"Historique reçu : {len(m1):,} minutes M1 gardées, dernière {m1.index[-1] if len(m1) else '-'}")
        return f"ok,{len(m1)}"

@app.post("/decide", response_class=PlainTextResponse)
async def decide_route(request: Request):
    check(request)
    body = (await request.body()).decode("ascii", errors="ignore")
    return await _run_in_thread(_decide_text, body)

async def _run_in_thread(fn, *a):
    import anyio
    return await anyio.to_thread.run_sync(fn, *a)

def _decide_text(body):
    lines = body.splitlines()
    try:
        rid, t_dec, pos, upnl, age, n_day = lines[0].split(",")[:6]
    except Exception:
        return "error,requete illisible"
    with LOCK:
        if S["m1"] is None:
            log("Requête reçue sans historique -> l'EA doit renvoyer l'historique")
            return "need_history"
        t0 = time.time()
        try:
            new = read_bars(lines[1:])
            m1 = S["m1"]
            if len(new) and new.index[0] - m1.index[-1] > pd.Timedelta(minutes=10):
                log(f"Historique incomplet (trou de {m1.index[-1]} à {new.index[0]}) -> l'EA renvoie l'historique")
                S["m1"] = None
                return "need_history"
            m1 = pd.concat([m1[~m1.index.isin(new.index)], new]).sort_index().iloc[-WINDOW - 5000:]
            S["m1"] = m1
            t = pd.to_datetime(int(t_dec), unit="s")
            state = dict(pos=int(pos), upnl=float(upnl), age=int(age), n_day=int(n_day))
            action, votes, info = decide(m1, t, state)
        except Exception as e:
            log(f"ERREUR : {e}\n{traceback.format_exc()}")
            return f"error,{str(e)[:120].replace(',', ';')}"
        finally:
            gc.collect()
        dt = time.time() - t0; S["secs"].append(dt); S["n_dec"] += 1
        trend = {1.0: "HAUSSE", -1.0: "BAISSE", 0.0: "aucune"}[info["allowed"]]
        S["last"] = f"{t:%Y-%m-%d %H:%M} UTC {NAMES[action]}"
        log(f"{t:%Y-%m-%d %H:%M} UTC | {NAMES[action]:6s} | comité {NAMES[info['raw']]} votes {votes} | tendance {trend} "
            f"| pos {state['pos']:+d} age {state['age']} | portée par : {info['why']} | close {info['close']:.2f} | {dt:.1f}s")
        return f"{rid},{action},{''.join(str(v) for v in votes)}"

log(f"Comité OR M1 chargé (agents {MODELS[4]} ; {len(MODELS[2])} features ; règle {TREND_COLS} ; fenêtre {WINDOW:,} min ; "
    f"token {'activé' if TOKEN else 'NON activé'})")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
