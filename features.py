"""Features du comité v4 calculées en direct — mêmes calculs que load() de research/scalp_rl_agent_v4.py.
Nécessite ohlc, ind, TF (research/scalp_phase1_map.py) et strategy_signals (research/scalp_strategies.py),
importés par le service ou insérés au-dessus sur Kaggle (où la parité avec load() est vérifiée).
m1 : DataFrame UTC, colonnes o, h, l, c (prix mid), spread ($), v (volume), index = ouverture de la minute."""
import numpy as np, pandas as pd

ADV_H = [5, 10, 15, 30, 60]

def _m5_base(m1):
    m5 = ohlc(m1, "5min")
    F, atr5 = ind(m5, "m5")
    lags = {}
    for k in range(1, 12):
        lags[f"m5_ret_l{k}"] = (F["m5_ret1"] / F["m5_atr_n"]).shift(k); lags[f"m5_range_l{k}"] = F["m5_range"].shift(k)
    return m5, pd.concat([F, pd.DataFrame(lags, index=F.index)], axis=1), atr5

def _add_htf(F, m1, m5):
    t_dec = m5.index + pd.Timedelta(minutes=5)
    for name, rule in TF.items():
        f, _ = ind(ohlc(m1, rule), name.lower()); f.index = f.index + pd.Timedelta(rule)
        F = pd.merge_asof(F.assign(_t=t_dec).sort_values("_t"), f.sort_index(), left_on="_t", right_index=True,
                          direction="backward").drop(columns="_t").set_index(m5.index)
    hr = t_dec.hour + t_dec.minute / 60
    F["h_sin"], F["h_cos"], F["dow"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24), t_dec.dayofweek
    return F

def advisor_features(m1):
    """Entrées des 5 conseillers LightGBM (phase 1)."""
    m5, F, _ = _m5_base(m1)
    F["spread"] = m5.spread
    return _add_htf(F, m1, m5).replace([np.inf, -np.inf], np.nan).astype(np.float32)

def agent_features(m1, adv, gs=None):
    """Observation brute de l'agent v4 (avant normalisation). adv : DataFrame adv_5..adv_60 (probabilités, index M5) ;
    gs : Series des probabilités GoldStark (index M5) ou None."""
    m5, F, atr5 = _m5_base(m1)
    F["spread"] = m5.spread / atr5
    F = _add_htf(F, m1, m5)
    adv = adv.reindex(F.index).copy()
    adv["adv_goldstark"] = gs.reindex(F.index) if gs is not None else np.nan
    F = pd.concat([F, adv - 0.5], axis=1).replace([np.inf, -np.inf], np.nan)
    m5v = m5.assign(v=m1.v.resample("5min").sum().reindex(m5.index).fillna(0))
    STR = strategy_signals(m1[["o", "h", "l", "c", "v"]], m5v).reindex(F.index).fillna(0)
    return pd.concat([F, STR], axis=1), m5.c, atr5

def policy_action(W, obs):
    """Action déterministe d'un agent PPO (MLP tanh 256x256) exporté en numpy."""
    x = np.tanh(obs @ W["w0"].T + W["b0"]); x = np.tanh(x @ W["w1"].T + W["b1"])
    return int(np.argmax(x @ W["wa"].T + W["ba"]))

def feature_groups(cols):
    """Regroupe les colonnes par source : unités de temps (M5, M15, H1...), conseillers, stratégies, autres."""
    g = {}
    for i, c in enumerate(cols):
        if c.startswith("adv_"): k = "conseillers"
        elif c[:1] in "sbh" and c[1:3].isdigit(): k = "stratégies"
        elif c.split("_")[0] in ("m5", "m15", "m30", "h1", "h4", "d1"): k = c.split("_")[0].upper()
        else: k = "autres"
        g.setdefault(k, []).append(i)
    return g

def explain(policies, x, st, groups, action):
    """Diagnostic : on refait voter le comité en neutralisant UN groupe d'informations à la fois (valeur moyenne).
    Les groupes dont la neutralisation change la décision sont ceux qui la « portent »."""
    drivers = []
    for name, idx in groups.items():
        x2 = x.copy(); x2[idx] = 0.0
        a2, _ = committee_action(policies, np.concatenate([x2, st]))
        if a2 != action: drivers.append(f"{name}->{('PLAT', 'BUY', 'SELL')[a2]}")
    return ", ".join(drivers) if drivers else "aucun groupe seul (décision partagée)"

def committee_action(policies, obs):
    """Vote majoritaire ; égalité -> à plat. Renvoie (action, votes)."""
    votes = [policy_action(W, obs) for W in policies]
    counts = np.bincount(votes, minlength=3); best = np.flatnonzero(counts == counts.max())
    return (int(best[0]) if len(best) == 1 else 0), votes
