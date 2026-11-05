"""Copie de ohlc, ind et TF de research/scalp_phase1_map.py (ne pas modifier ici : mêmes calculs qu'à l'entraînement)."""
import numpy as np, pandas as pd

TF = {"M15": "15min", "H1": "1h", "H4": "4h", "D1": "1D"}

def ohlc(m1, rule):
    return (m1.resample(rule).agg({"o": "first", "h": "max", "l": "min", "c": "last", "spread": "mean"}).dropna())

def ind(b, p):
    """Indicateurs stationnaires d'une série de bougies (préfixe p)."""
    o, h, l, c = b.o, b.h, b.l, b.c
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26; hist = macd - macd.ewm(span=9, adjust=False).mean()
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    upm, dnm = h.diff(), -l.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), index=b.index).ewm(alpha=1 / 14, adjust=False).mean()
    ndm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=b.index).ewm(alpha=1 / 14, adjust=False).mean()
    pdi, ndi = 100 * pdm / atr, 100 * ndm / atr
    adx = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)).ewm(alpha=1 / 14, adjust=False).mean()
    f = pd.DataFrame({
        f"{p}_ret1": np.log(c / pc), f"{p}_ret3": np.log(c / c.shift(3)),
        f"{p}_range": (h - l) / atr, f"{p}_body": (c - o) / atr,
        f"{p}_wick_up": (h - np.maximum(o, c)) / atr, f"{p}_wick_dn": (np.minimum(o, c) - l) / atr,
        f"{p}_atr_n": atr / c, f"{p}_atr_z": (atr / c) / (atr / c).rolling(200).mean() - 1,
        f"{p}_rsi": 100 - 100 / (1 + up / dn.replace(0, np.nan)),
        f"{p}_d_ema20": (c - c.ewm(span=20, adjust=False).mean()) / atr,
        f"{p}_d_ema50": (c - c.ewm(span=50, adjust=False).mean()) / atr,
        f"{p}_macd_h": hist / atr, f"{p}_macd_h_d": hist.diff() / atr,
        f"{p}_bb": (c - ma20) / (2 * sd20), f"{p}_adx": adx, f"{p}_di": pdi - ndi,
    }, index=b.index)
    return f, atr

