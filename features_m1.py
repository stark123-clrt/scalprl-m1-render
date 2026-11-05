"""Features du comité OR M1 (suivi de tendance) — mêmes calculs que load_m1_agent() de research/scalp_rl_agent_m1.py,
appliqués à la fenêtre M1 reçue de l'EA (prix mid). Parité vérifiée sur Kaggle : research/scalp_m1_live_export.py.
Informations : indicateurs M1 + 10 dernières bougies, M5 / M15 / H1 / H4 FERMÉES, spread, heure, 16 stratégies M1
(+ leur bilan des 5 dernières minutes). Règle de tendance : features_v5.allowed_direction / strict_rule (EMA50 M15)."""
import numpy as np, pandas as pd
try:
    from phase1 import ohlc, ind, TF
    from strategies_m1 import signals
except ImportError:                       # sur Kaggle : fonctions déjà définies au-dessus
    pass

def agent_features_m1(m1):
    STR = signals(m1[["o", "h", "l", "c", "v"]])
    F, atr1 = ind(m1, "m1")
    lags = {}
    for k in range(1, 11):
        lags[f"m1_ret_l{k}"] = (F["m1_ret1"] / F["m1_atr_n"]).shift(k); lags[f"m1_range_l{k}"] = F["m1_range"].shift(k)
    F = pd.concat([F, pd.DataFrame(lags, index=F.index)], axis=1).astype(np.float32)
    F["spread"] = (m1.spread / atr1).astype(np.float32)
    t_dec = m1.index + pd.Timedelta(minutes=1)
    for name, rule in {"M5": "5min", **TF}.items():
        if name == "D1": continue
        f, _ = ind(ohlc(m1, rule), name.lower()); f.index = f.index + pd.Timedelta(rule)
        f = f.astype(np.float32).sort_index()
        pos = np.searchsorted(f.index.values, t_dec.values, side="right") - 1          # dernière bougie FERMÉE
        vals = f.values[np.clip(pos, 0, None)]; vals[pos < 0] = np.nan
        F = pd.concat([F, pd.DataFrame(vals, index=F.index, columns=f.columns)], axis=1)
    hr = t_dec.hour + t_dec.minute / 60
    F["h_sin"], F["h_cos"], F["dow"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24), t_dec.dayofweek
    STR = STR.reindex(F.index).fillna(0).astype(np.float32)
    rec = STR.rolling(5, min_periods=1).sum().clip(-1, 1); rec.columns = [f"{c}_r5" for c in STR.columns]
    F = pd.concat([F, STR, rec], axis=1).replace([np.inf, -np.inf], np.nan)
    return F, m1.c, atr1

def feature_groups_m1(cols):
    """Regroupe les colonnes par source (M1, M5, M15, H1, H4, stratégies, autres) pour le diagnostic du journal."""
    g = {}
    for i, c in enumerate(cols):
        if c[:1] == "y" and c[1:3].isdigit(): k = "stratégies"
        elif c.split("_")[0] in ("m1", "m5", "m15", "h1", "h4"): k = c.split("_")[0].upper()
        else: k = "autres"
        g.setdefault(k, []).append(i)
    return g
