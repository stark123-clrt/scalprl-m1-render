"""SCALPING M5 — Agent RL (PPO) V4 : V3 + 21 stratégies or M5 multi-timeframes comme outils.
V3 : scalpeur 60 min max, coût progressif, années récentes favorisées,
GoldStark walk-forward comme 6e avis du conseiller, plusieurs graines (vote ensuite).

Aucune règle imposée (pas de filtre de session ni de régime) : l'agent reçoit TOUTES les informations et décide seul.
  Observation à chaque clôture M5 :
    - ~130 indicateurs multi-timeframes (M5 + M15/H1/H4/D1 fermées), standardisés sur la période d'entraînement ;
    - le « conseiller » : probabilités du modèle supervisé walk-forward à 5/10/15/30/60 min (phase 1, hors échantillon) ;
    - son état : position (-1/0/+1), gain latent en ATR, nombre de bougies en position.
  Actions : 0 = à plat, 1 = acheteur, 2 = vendeur (position cible ; entrer / sortir / se retourner à chaque bougie).
  Récompense (en unités d'ATR M5) : variation du gain latent - coût à chaque changement de position
    - petite pénalité d'inactivité (pousse à trader souvent, sans l'y obliger).
Entraînement : PPO (stable-baselines3) sur des semaines tirées au hasard en 2018-2023 ; choix du meilleur entraînement
sur 2024 ; TEST final sur 2025-2026 (jamais vu). Plusieurs graines.
Évaluation à plusieurs niveaux de coût (0,05 / 0,15 / 0,25 / 0,35 $ par trade et par once).
"""
import json, os, subprocess, sys, time
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "stable-baselines3>=2.3", "gymnasium"], check=False)
import numpy as np, pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

INPUT = Path("/kaggle/input"); OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
TRAIN = ("2018-01-01", "2024-01-01"); VAL = ("2024-01-01", "2025-01-01"); TEST = ("2025-01-01", "2027-01-01")
COST_TRAIN = float(os.environ.get("GS_COST", 0.15))           # $ par once et par trade aller-retour
IDLE_PEN = float(os.environ.get("GS_IDLE", 0.002))            # pénalité par bougie à plat (en ATR) -> agressivité
STEPS = int(os.environ.get("GS_STEPS", 12_000_000))
SEEDS = [int(s) for s in os.environ.get("GS_SEEDS", "1,2").split(",")]
EP_LEN = 2016                                                  # ~1 semaine de bougies M5

# ── Données : fonctions de la phase 1 (ohlc, ind, TF) insérées au-dessus lors de l'envoi sur Kaggle ──

def load():
    m = pd.read_parquet(next(INPUT.rglob("xau_m1_bidask.parquet"))).set_index("datetime").sort_index()
    m1 = pd.DataFrame({"o": (m.o_bid + m.o_ask) / 2, "h": (m.h_bid + m.h_ask) / 2, "l": (m.l_bid + m.l_ask) / 2,
                       "c": (m.c_bid + m.c_ask) / 2, "spread": m.spread, "v": m.v})
    m5 = ohlc(m1, "5min")
    F, atr5 = ind(m5, "m5")
    lags = {}
    for k in range(1, 12):
        lags[f"m5_ret_l{k}"] = (F["m5_ret1"] / F["m5_atr_n"]).shift(k); lags[f"m5_range_l{k}"] = F["m5_range"].shift(k)
    F = pd.concat([F, pd.DataFrame(lags, index=F.index)], axis=1); F["spread"] = m5.spread / atr5   # en ATR : comparable entre années
    t_dec = m5.index + pd.Timedelta(minutes=5)
    for name, rule in TF.items():
        f, _ = ind(ohlc(m1, rule), name.lower()); f.index = f.index + pd.Timedelta(rule)
        F = pd.merge_asof(F.assign(_t=t_dec).sort_values("_t"), f.sort_index(), left_on="_t", right_index=True,
                          direction="backward").drop(columns="_t").set_index(m5.index)
    hr = t_dec.hour + t_dec.minute / 60
    F["h_sin"], F["h_cos"], F["dow"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24), t_dec.dayofweek
    adv = pd.read_parquet(next(INPUT.rglob("scalp_preds_wf.parquet"))).reindex(F.index)
    adv.columns = [f"adv_{c}" for c in adv.columns]
    gsp = list(INPUT.rglob("wf_probs.parquet"))
    if gsp:
        g = pd.read_parquet(gsp[0]); g = g[g.part == "test"][["datetime", "p"]].copy()
        g["datetime"] = pd.to_datetime(g["datetime"])
        if g["datetime"].dt.tz is not None: g["datetime"] = g["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
        g["t_avail"] = g["datetime"] + pd.Timedelta(minutes=15)          # connue à la CLÔTURE de la bougie M15
        g = g.sort_values("t_avail")
        td = pd.DataFrame({"t": t_dec.tz_localize(None) if t_dec.tz is not None else t_dec}).sort_values("t")
        gm = pd.merge_asof(td, g[["t_avail", "p"]], left_on="t", right_on="t_avail", direction="backward",
                           tolerance=pd.Timedelta(minutes=30))
        adv["adv_goldstark"] = gm["p"].values
    F = pd.concat([F, adv - 0.5], axis=1).replace([np.inf, -np.inf], np.nan)
    m5v = m5.assign(v=m1.v.resample("5min").sum().reindex(m5.index).fillna(0))
    STR = strategy_signals(m1[["o", "h", "l", "c", "v"]], m5v).reindex(F.index).fillna(0)
    print("  stratégies (signaux/jour moyens) : " + ", ".join(
        f"{c[:3]} {np.mean(STR[c] != 0) * 276:.1f}" for c in STR.columns if not c.endswith("_r3")), flush=True)
    F = pd.concat([F, STR], axis=1)
    price = m5.c.values; atr = atr5.values
    gap = np.r_[np.diff(m5.index.values) > np.timedelta64(15, "m"), True]          # week-end / trou après la bougie
    return F, price, atr, gap, t_dec


# ════════════════════════ VERSION 2 : un vrai scalpeur ════════════════════════
MAX_HOLD = int(os.environ.get("GS_MAX_HOLD", 12))           # 12 bougies M5 = 1 h max par position (fermeture forcée)
HOLD_PEN = float(os.environ.get("GS_HOLD_PEN", 0.01))       # pénalité qui grandit avec la durée de la position (ATR)
TARGET = int(os.environ.get("GS_TARGET", 20))               # minimum DOUX de trades par jour
FREQ_PEN = float(os.environ.get("GS_FREQ_PEN", 0.1))        # pénalité PAR TRADE MANQUANT en fin de journée (ATR) :
                                                            # > coût d'un trade, sinon ne pas trader serait plus rentable
MIRROR = os.environ.get("GS_MIRROR", "1") == "1"            # marché miroir : hausses et baisses équilibrées
CURRICULUM = [float(x) for x in os.environ.get("GS_CURRICULUM", "0,0,0,0.05,0.05,0.05,0.15,0.15,0.15,0.15").split(",")]
RECENCY = float(os.environ.get("GS_RECENCY", 0.35))       # poids des années récentes : exp(RECENCY x (année - 2018))

def mirror_features(F):
    """Même marché vu à l'envers : les features de direction changent de signe, le reste ne bouge pas."""
    M = F.copy()
    for c in F.columns:
        if any(k in c for k in ("_ret1", "_ret3", "_body", "_d_ema", "_macd_h", "_bb", "_di", "m5_ret_l")) or c.startswith("adv_")                 or (len(c) > 3 and c[0] == "s" and c[1:3].isdigit() and c[3] == "_"):   # signaux des stratégies : BUY <-> SELL
            M[c] = -F[c]
        elif c.endswith("_rsi"):
            M[c] = 100 - F[c]
    for c in F.columns:                                      # mèche haute <-> mèche basse
        if c.endswith("_wick_up"): M[c] = F[c.replace("_wick_up", "_wick_dn")]
        if c.endswith("_wick_dn"): M[c] = F[c.replace("_wick_dn", "_wick_up")]
    return M

class ScalpEnv(gym.Env):
    """Une position à la fois, 1 h maximum, rien la nuit / le week-end, objectif de trades par jour."""
    def __init__(self, Xs, prices, atr, brk, day, lo, hi, cost, random_start=True, ep_len=EP_LEN):
        super().__init__()
        self.Xs, self.ps, self.atr, self.brk, self.day = Xs, prices, atr, brk, day
        self.years = YEARS_ARR
        self.lo, self.hi, self.cost, self.rs, self.ep_len = lo, hi, cost, random_start, ep_len
        self.observation_space = spaces.Box(-10, 10, (Xs[0].shape[1] + 4,), np.float32)
        self.action_space = spaces.Discrete(3)
    def _obs(self):
        st = np.array([self.pos, np.clip(self.upnl / max(self.atr[self.t], 1e-6), -10, 10), self.age / MAX_HOLD,
                       min(self.n_day / max(TARGET, 1), 3)], np.float32)
        return np.concatenate([self.X[self.t], st])
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        k = int(self.np_random.integers(len(self.Xs))) if self.rs else 0       # monde normal ou miroir
        self.X, self.p = self.Xs[k], self.ps[k]
        if self.rs:
            if getattr(self, "cw", None) is None:
                yrs = self.years[self.lo:max(self.lo + 1, self.hi - self.ep_len)]
                w = np.exp(RECENCY * (yrs - 2018)); self.cw = np.cumsum(w) / w.sum()
            self.t = self.lo + int(np.searchsorted(self.cw, self.np_random.random()))
        else:
            self.t = self.lo
        self.t_end = min(self.t + self.ep_len, self.hi - 1) if self.rs else self.hi - 1
        self.pos, self.entry, self.upnl, self.age, self.n_day = 0, 0.0, 0.0, 0, 0
        self.trades, self.sides, self.holds = [], [], []
        return self._obs(), {}
    def _close(self):
        self.trades.append(self.pos * (self.p[self.t] - self.entry) - self.cost)
        self.sides.append(self.pos); self.holds.append(self.age)
        self.pos, self.age = 0, 0
    def step(self, a):
        target = (0, 1, -1)[int(a)]
        if self.brk[self.t] or self.age >= MAX_HOLD:          # pause journalière / week-end / 1 h atteinte -> à plat
            target = 0
        a_t = max(self.atr[self.t], 1e-6); r = 0.0
        if target != self.pos:
            r -= (self.cost / 2) * abs(target - self.pos) / a_t
            if self.pos != 0: self._close()
            if target != 0:
                self.pos, self.entry, self.age = target, self.p[self.t], 0
                self.n_day += 1
        nxt = self.t + 1
        r += self.pos * (self.p[nxt] - self.p[self.t]) / a_t
        if self.pos != 0:
            self.age += 1
            r -= HOLD_PEN * self.age / MAX_HOLD                   # garder longtemps coûte de plus en plus
        if self.day[nxt] != self.day[self.t]:                     # fin de journée : objectif de fréquence
            if self.n_day < TARGET: r -= FREQ_PEN * (TARGET - self.n_day)
            self.n_day = 0
        self.t = nxt
        self.upnl = self.pos * (self.p[self.t] - self.entry) if self.pos else 0.0
        done = self.t >= self.t_end
        if done and self.pos != 0: self._close()
        return self._obs(), float(r), done, False, {}

def evaluate(model, X, price, atr, brk, day, lo, hi, t_dec):
    env = ScalpEnv([X], [price], atr, brk, day, lo, hi, 0.0, random_start=False)
    obs, _ = env.reset(); done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(a)
    g = np.array(env.trades); sd = np.array(env.sides); ho = np.array(env.holds)
    days = max(len(np.unique(day[lo:hi])), 1)
    res = {"trades/jour": round(len(g) / days, 1),
           "duree_moy_min": round(5 * ho.mean(), 1) if len(ho) else 0,
           "part_BUY": f"{np.mean(sd == 1):.0%}" if len(sd) else "-",
           "brut_BUY": round(g[sd == 1].sum(), 0) if len(g) else 0, "brut_SELL": round(g[sd == -1].sum(), 0) if len(g) else 0,
           "buy_and_hold": round(price[hi - 1] - price[lo], 0)}
    for c in (0.05, 0.15, 0.25, 0.35):
        v = g - c; gl = -v[v < 0].sum(); eq = np.cumsum(v) if len(v) else np.zeros(1)
        res[f"cout {c}$"] = dict(reussite=f"{np.mean(v > 0):.1%}" if len(v) else "-", PF=round(v[v > 0].sum() / gl, 3) if gl else None,
                                 gain_moy=round(v.mean(), 3) if len(v) else 0, total=round(v.sum(), 0),
                                 max_dd=round((np.maximum.accumulate(eq) - eq).max(), 0))
    return res

if __name__ == "__main__":
    t0 = time.time()
    F, price, atr, gap, t_dec = load()
    idx = lambda a, b: (int(np.searchsorted(t_dec, pd.Timestamp(a))), int(np.searchsorted(t_dec, pd.Timestamp(b))))
    tr_lo, tr_hi = idx(*TRAIN); va_lo, va_hi = idx(*VAL); te_lo, te_hi = idx(*TEST)
    Fm = mirror_features(F)
    both = np.vstack([F.values[tr_lo:tr_hi], Fm.values[tr_lo:tr_hi]])
    mu, sd = np.nanmean(both, 0), np.nanstd(both, 0) + 1e-9          # normalisation symétrique
    std = lambda A: np.clip(np.nan_to_num((A - mu) / sd), -10, 10).astype(np.float32)
    X, Xm = std(F.values), std(Fm.values)
    day = t_dec.normalize().values.astype("datetime64[D]").astype(np.int64)
    global YEARS_ARR
    YEARS_ARR = np.asarray(t_dec.year)
    brk = gap.copy()                                                  # dernière bougie avant une pause (nuit / week-end)
    Xs, Ps = ([X, Xm], [price, -price]) if MIRROR else ([X], [price])
    print(f"V4 | {len(F):,} bougies M5 x {F.shape[1]} | objectif {TARGET} trades/j | max {MAX_HOLD*5} min | "
          f"miroir {MIRROR} ({(time.time()-t0)/60:.1f} min)", flush=True)

    results = []
    for seed in SEEDS:
        venv = DummyVecEnv([lambda: ScalpEnv(Xs, Ps, atr, brk, day, tr_lo, tr_hi, COST_TRAIN) for _ in range(16)])
        model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=4096, n_epochs=5, learning_rate=2e-4,
                    gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])), verbose=0, device="cpu")
        best, chunk = None, STEPS // 10
        for k in range(10):
            venv.set_attr("cost", CURRICULUM[min(k, len(CURRICULUM) - 1)])   # coût progressif (0 -> 0,05 -> 0,15 $)
            model.learn(chunk, reset_num_timesteps=False)
            val = evaluate(model, X, price, atr, brk, day, va_lo, va_hi, t_dec)
            c = val[f"cout {COST_TRAIN}$"]
            print(f"  graine {seed} | {(k+1)*chunk/1e6:.1f}M pas | VAL 2024 : {val['trades/jour']} trades/j, {val['duree_moy_min']} min, "
                  f"BUY {val['part_BUY']} (brut BUY {val['brut_BUY']} / SELL {val['brut_SELL']}) | coût {COST_TRAIN}$ -> {c} "
                  f"| buy&hold {val['buy_and_hold']} ({(time.time()-t0)/60:.0f} min)", flush=True)
            if best is None or c["total"] > best[0]:
                best = (c["total"], k); model.save(OUT / f"ppo_v4_seed{seed}_best")
        results.append((best[0], seed))
        json.dump({"seed": seed, "val_total_015": float(best[0]), "chunk": int(best[1])},
                  open(OUT / f"ppo_v4_seed{seed}_val.json", "w"))
    results.sort(reverse=True)
    print("\n" + "=" * 100 + f"\nTEST FINAL 2025-2026 (jamais vu) — objectif {TARGET} trades/j\n" + "=" * 100)
    for _, s in results:
        r = evaluate(PPO.load(OUT / f"ppo_v4_seed{s}_best"), X, price, atr, brk, day, te_lo, te_hi, t_dec)
        print(f"\n  graine {s} :")
        for k, v in r.items(): print(f"    {k} : {v}")
    print(f"\nTerminé en {(time.time()-t0)/60:.1f} min")
