"""Features du comité M5 v5 STRICT (suivi de tendance) — mêmes calculs que load() de la v4 SANS GoldStark,
avec les 29 stratégies (21 + 8 de tendance). Règle de tendance : entrée seulement dans le sens où H1 et M15 sont alignés.
Utilise _m5_base / _add_htf de features.py ; strategy_signals = version tendance (strategies_trend.py)."""
import numpy as np, pandas as pd
try:
    from features import _m5_base, _add_htf
except ImportError:                       # sur Kaggle : fonctions déjà définies au-dessus
    pass

def agent_features_v5(m1, adv):
    m5, F, atr5 = _m5_base(m1)
    F["spread"] = m5.spread / atr5
    F = _add_htf(F, m1, m5)
    F = pd.concat([F, adv.reindex(F.index) - 0.5], axis=1).replace([np.inf, -np.inf], np.nan)
    m5v = m5.assign(v=m1.v.resample("5min").sum().reindex(m5.index).fillna(0))
    STR = strategy_signals(m1[["o", "h", "l", "c", "v"]], m5v).reindex(F.index).fillna(0)
    return pd.concat([F, STR], axis=1), m5.c, atr5

def allowed_direction(h1_d_ema50, m15_d_ema50):
    """+1 / -1 si H1 et M15 sont alignés (prix H1 à plus de 0,3 ATR de son EMA50), sinon 0."""
    h1, m15 = float(np.nan_to_num(h1_d_ema50)), float(np.nan_to_num(m15_d_ema50))
    return float(np.sign(h1)) if abs(h1) > 0.3 and np.sign(h1) == np.sign(m15) else 0.0

def strict_rule(action, pos, allowed):
    """Applique la règle stricte à l'action du comité (0 à plat, 1 BUY, 2 SELL), comme à l'entraînement."""
    tgt = (0, 1, -1)[action]
    if pos != 0 and allowed == -pos: tgt = 0
    if tgt != 0 and tgt != pos and tgt != allowed: tgt = 0
    return {0: 0, 1: 1, -1: 2}[tgt]
