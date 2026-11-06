"""OR M1 — STRATÉGIES DE TENDANCE VUES SUR YOUTUBE / TRADINGVIEW (demande de l'utilisateur, 03/11/2026).

Étape 1 avant le comité M1 : on code 16 stratégies de tendance M1 trouvées en ligne, et on mesure chacune honnêtement
sur l'or M1 (Dukascopy, 2016 -> 2026) :
  - signal à la CLÔTURE d'une bougie M1, entrée à l'ouverture de la suivante (prix moyen bid/ask) ;
  - sortie : stop 1,5 ATR, objectif 2 ATR, ou fermeture après 60 min (si stop et objectif touchés dans la même
    minute, on compte le stop) ; une seule position à la fois par stratégie ;
  - coût 0,15 $ par trade (comme les tests précédents), et 0,30 $ en test de résistance ;
  - unités supérieures (M5 / M15) : seulement des bougies FERMÉES ;
  - marché miroir (prix inversés) pour vérifier que ça marche aussi en baisse.
Résultats en $ pour 1 once (0,01 lot). Sortie : m1_trend_signals.parquet (signaux pour le futur comité M1).
"""
import time
from pathlib import Path
import numpy as np, pandas as pd

INPUT = Path("/kaggle/input"); OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
SL_K, TP_K, HOLD = 1.5, 2.0, 60
COSTS = (0.15, 0.30)

# ─────────────────────────────── indicateurs ───────────────────────────────
ema = lambda s, n: s.ewm(span=n, adjust=False).mean()

def rsi(c, n):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def supertrend(h, l, c, n=10, k=3.0):
    a = atr(h, l, c, n).values; hl2 = ((h + l) / 2).values; cc = c.values
    up, dn = hl2 - k * a, hl2 + k * a
    d = np.ones(len(cc)); fu, fd = up.copy(), dn.copy()
    for i in range(1, len(cc)):
        fu[i] = max(up[i], fu[i - 1]) if cc[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if cc[i - 1] < fd[i - 1] else dn[i]
        d[i] = 1 if cc[i] > fd[i - 1] else (-1 if cc[i] < fu[i - 1] else d[i - 1])
    return pd.Series(d, index=c.index)

def psar(h, l, step=0.02, mx=0.2):
    hh, ll = h.values, l.values; n = len(hh); d = np.ones(n); sar = ll[0]; ep = hh[0]; af = step; up = True
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, ll[i - 1], ll[i - 2] if i > 1 else ll[i - 1])
            if ll[i] < sar: up, sar, ep, af = False, ep, ll[i], step
            elif hh[i] > ep: ep, af = hh[i], min(af + step, mx)
        else:
            sar = max(sar, hh[i - 1], hh[i - 2] if i > 1 else hh[i - 1])
            if hh[i] > sar: up, sar, ep, af = True, ep, hh[i], step
            elif ll[i] < ep: ep, af = ll[i], min(af + step, mx)
        d[i] = 1 if up else -1
    return pd.Series(d, index=h.index)

def adx_di(h, l, c, n=14):
    a = atr(h, l, c, n); upm, dnm = h.diff(), -l.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), index=h.index).ewm(alpha=1 / n, adjust=False).mean()
    ndm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=h.index).ewm(alpha=1 / n, adjust=False).mean()
    pdi, ndi = 100 * pdm / a, 100 * ndm / a
    return (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)).ewm(alpha=1 / n, adjust=False).mean(), pdi, ndi

def htf(df, rule, fn):
    """Indicateur calculé sur une unité supérieure FERMÉE, ramené sur l'index M1 (décision à la clôture M1)."""
    b = df.resample(rule).agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    x = fn(b)
    x.index = x.index + pd.Timedelta(rule)                       # disponible à la clôture de la bougie supérieure
    return x.reindex(df.index + pd.Timedelta("1min"), method="ffill").set_axis(df.index)

cross_up = lambda a, b: (a > b) & (a.shift(1) <= b.shift(1))
cross_dn = lambda a, b: (a < b) & (a.shift(1) >= b.shift(1))

def signals(df):
    o, h, l, c, v = df.o, df.h, df.l, df.c, df.v
    S = {}
    def both(name, buy, sell): S[name] = np.where(buy.fillna(False), 1, np.where(sell.fillna(False), -1, 0)).astype(np.int8)
    e5, e9, e13, e21, e34, e50 = (ema(c, n) for n in (5, 9, 13, 21, 34, 50))
    e100, e200 = ema(c, 100), ema(c, 200)
    r14, r7 = rsi(c, 14), rsi(c, 7)
    # y01 EMA 9/21 + RSI 14 (vidéo « 1 MIN Gold Scalping EMA 9.21 + RSI »)
    both("y01_ema9_21_rsi", cross_up(e9, e21) & (r14 > 50), cross_dn(e9, e21) & (r14 < 50))
    # y02 EMA 9/21/34/50 empilées (vidéo « Top 4 ema 1 Min Gold Scalping ») : entrée au début de l'alignement
    st_up = (e9 > e21) & (e21 > e34) & (e34 > e50); st_dn = (e9 < e21) & (e21 < e34) & (e34 < e50)
    both("y02_ema_stack_4", st_up & ~st_up.shift(1, fill_value=False), st_dn & ~st_dn.shift(1, fill_value=False))
    # y03 EMA 9>21 + Stochastique RSI qui remonte de la zone < 20 (pullback)
    rr = r14; srsi = 100 * (rr - rr.rolling(14).min()) / (rr.rolling(14).max() - rr.rolling(14).min()).replace(0, np.nan)
    k = srsi.rolling(3).mean(); d = k.rolling(3).mean()
    both("y03_ema_stochrsi", (e9 > e21) & cross_up(k, d) & (k.shift(1) < 20), (e9 < e21) & cross_dn(k, d) & (k.shift(1) > 80))
    # y04 Croisement de moyennes + SAR parabolique, seulement avec la tendance (EMA 200)
    sar = psar(h, l)
    both("y04_ma_cross_psar", cross_up(e9, e21) & (sar > 0) & (c > e200), cross_dn(e9, e21) & (sar < 0) & (c < e200))
    # y05 EMA 5/13 + RSI 7 (script TradingView « aggressive scalping » M1/M3)
    both("y05_ema5_13_rsi7", cross_up(e5, e13) & (r7 > 50), cross_dn(e5, e13) & (r7 < 50))
    # y06 Hiérarchie EMA 80/90 vs 340/500 : le prix recroise l'EMA 80 dans le sens de la tendance lente
    e80, e90, e340, e500 = (ema(c, n) for n in (80, 90, 340, 500))
    both("y06_ema_80_90_340_500", (e80 > e340) & (e90 > e500) & cross_up(c, e80), (e80 < e340) & (e90 < e500) & cross_dn(c, e80))
    # y07 Tendance EMA 200 + pullback qui touche l'EMA 100 puis referme du bon côté
    both("y07_ema200_pullback100", (c > e200) & (l <= e100) & (c > e100) & (e100 > e200),
         (c < e200) & (h >= e100) & (c < e100) & (e100 < e200))
    # y08 EA MQL5 : EMA M15 (tendance) + Supertrend M1 + Stochastique qui sort de survente/surachat
    t15 = htf(df, "15min", lambda b: np.sign(b.c - ema(b.c, 50)))
    stm1 = supertrend(h, l, c)
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    sk = (100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)).rolling(3).mean(); sd = sk.rolling(3).mean()
    both("y08_m15ema_supertrend_stoch", (t15 > 0) & (stm1 > 0) & cross_up(sk, sd) & (sd < 30),
         (t15 < 0) & (stm1 < 0) & cross_dn(sk, sd) & (sd > 70))
    # y09 VWAP + MACD 12/26/9 (FXOpen « 1-minute gold scalping »)
    day = c.index.normalize(); tp = (h + l + c) / 3; vv = v.clip(lower=1e-9)
    vwap = (tp * vv).groupby(day).cumsum() / vv.groupby(day).cumsum()
    macd = ema(c, 12) - ema(c, 26); sig = ema(macd, 9)
    both("y09_vwap_macd", (c > vwap) & cross_up(macd, sig), (c < vwap) & cross_dn(macd, sig))
    # y10 VWAP + Supertrend + RSI (script TradingView) : retournement du Supertrend du côté du VWAP
    both("y10_vwap_supertrend_rsi", (c > vwap) & (stm1 > 0) & (stm1.shift(1) < 0) & (r14 > 50),
         (c < vwap) & (stm1 < 0) & (stm1.shift(1) > 0) & (r14 < 50))
    # y11 Heikin Ashi : 2 bougies vertes sans mèche basse après une rouge, dans la tendance EMA 200
    hc = (o + h + l + c) / 4; ho = hc.to_numpy(copy=True); hcv = hc.to_numpy(); ov = o.to_numpy()
    ho[0] = (ov[0] + c.values[0]) / 2
    for i in range(1, len(ho)): ho[i] = (ho[i - 1] + hcv[i - 1]) / 2
    ho = pd.Series(ho, index=c.index); hh_ = pd.concat([h, ho, hc], axis=1).max(axis=1); hl_ = pd.concat([l, ho, hc], axis=1).min(axis=1)
    g, r = hc > ho, hc < ho
    nolow, nohigh = (pd.concat([ho, hc], axis=1).min(axis=1) - hl_) < 1e-9, (hh_ - pd.concat([ho, hc], axis=1).max(axis=1)) < 1e-9
    both("y11_heikin_ashi", (c > e200) & g & g.shift(1) & r.shift(2) & nolow, (c < e200) & r & r.shift(1) & g.shift(2) & nohigh)
    # y12 Cassure de Donchian 20 avec filtre EMA 200
    both("y12_donchian20_ema200", (c > e200) & (c > h.shift(1).rolling(20).max()), (c < e200) & (c < l.shift(1).rolling(20).min()))
    # y13 MACD qui passe au-dessus de zéro, avec filtre EMA 200
    both("y13_macd_zero_ema200", (c > e200) & cross_up(macd, macd * 0), (c < e200) & cross_dn(macd, macd * 0))
    # y14 ADX > 25 + croisement DI, tendance M5 (EMA 50)
    adx, pdi, ndi = adx_di(h, l, c)
    t5 = htf(df, "5min", lambda b: np.sign(b.c - ema(b.c, 50)))
    both("y14_adx_di_m5", (adx > 25) & (t5 > 0) & cross_up(pdi, ndi), (adx > 25) & (t5 < 0) & cross_up(ndi, pdi))
    # y15 Cassure de Bollinger (20, 2) avec EMA 50 > EMA 200
    mb, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    both("y15_bollinger_break", (e50 > e200) & cross_up(c, mb + 2 * sd20), (e50 < e200) & cross_dn(c, mb - 2 * sd20))
    # y16 Double Supertrend : retournement M1 dans le sens du Supertrend M5
    st5 = htf(df, "5min", lambda b: supertrend(b.h, b.l, b.c))
    both("y16_double_supertrend", (st5 > 0) & (stm1 > 0) & (stm1.shift(1) < 0), (st5 < 0) & (stm1 < 0) & (stm1.shift(1) > 0))
    return pd.DataFrame(S, index=df.index)

# ─────────────────────────────── simulation ───────────────────────────────
def simulate(sig, o, h, l, c, a):
    """Renvoie (i_entrée, pnl brut en $ par once) pour chaque trade, positions non superposées."""
    n = len(c); idx = np.flatnonzero(sig[: n - HOLD - 2] != 0)
    if len(idx) == 0: return np.array([], int), np.array([])
    out_i, out_p, last_exit = [], [], -1
    for s in range(0, len(idx), 50_000):
        E = idx[s:s + 50_000]; side = sig[E].astype(float)
        ent = o[E + 1]; sl = SL_K * a[E]; tp = TP_K * a[E]
        rows = E[:, None] + 1 + np.arange(HOLD)[None, :]
        H, L = h[rows], l[rows]
        fav = np.where(side[:, None] > 0, H - ent[:, None], ent[:, None] - L)        # mouvement favorable
        adv = np.where(side[:, None] > 0, ent[:, None] - L, H - ent[:, None])        # mouvement défavorable
        hit_sl, hit_tp = adv >= sl[:, None], fav >= tp[:, None]
        first_sl = np.where(hit_sl.any(1), hit_sl.argmax(1), HOLD); first_tp = np.where(hit_tp.any(1), hit_tp.argmax(1), HOLD)
        ex = np.minimum(first_sl, first_tp)
        pnl = np.where(first_sl <= first_tp, np.where(first_sl < HOLD, -sl, np.nan), tp)
        timeout = np.isnan(pnl)
        pnl[timeout] = side[timeout] * (c[E[timeout] + HOLD] - ent[timeout])
        ex = np.where(timeout, HOLD - 1, ex) + E + 1
        for e, x, p in zip(E, ex, pnl):
            if e > last_exit:
                out_i.append(e); out_p.append(p); last_exit = x
    return np.array(out_i), np.array(out_p)

def summarize(ii, pp, years, ndays, lo, hi, cost):
    m = (years[ii] >= lo) & (years[ii] <= hi) if len(ii) else np.array([], bool)
    p = pp[m] - cost if len(ii) else np.array([])
    days = ndays.loc[lo:hi].sum()
    if len(p) == 0: return dict(tpj=0.0, reussite="-", PF=0.0, total=0)
    g, ls = p[p > 0].sum(), -p[p < 0].sum()
    return dict(tpj=round(len(p) / max(days, 1), 1), reussite=f"{100 * (p > 0).mean():.0f}%", PF=round(g / ls, 3) if ls else 99.0, total=round(p.sum()))

if __name__ == "__main__":
    t0 = time.time()
    m = pd.read_parquet(next(INPUT.rglob("xau_m1_bidask.parquet")))
    df = pd.DataFrame({k: (m[f"{k}_bid"] + m[f"{k}_ask"]).values / 2 for k in "ohlc"}, index=pd.DatetimeIndex(m.datetime))
    df["v"] = m.v.values
    df = df[~df.index.duplicated()].sort_index()
    print(f"or M1 : {len(df):,} bougies, {df.index[0]} -> {df.index[-1]} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    years = df.index.year.values
    ndays = pd.Series(1, index=pd.Index(df.index.normalize().unique())).groupby(lambda d: d.year).sum()

    results = {}
    for mode in ("REEL", "MIROIR"):
        d = df if mode == "REEL" else pd.DataFrame({"o": -df.o, "h": -df.l, "l": -df.h, "c": -df.c, "v": df.v})
        S = signals(d)
        if mode == "REEL":
            S.to_parquet(OUT / "m1_trend_signals.parquet")
            print("signaux par jour (moyenne) : " + " | ".join(f"{k[:3]} {(S[k] != 0).sum() / ndays.sum():.0f}" for k in S), flush=True)
        a = atr(d.h, d.l, d.c).values
        for k in S:
            ii, pp = simulate(S[k].values, d.o.values, d.h.values, d.l.values, d.c.values, a)
            results[(mode, k)] = (ii, pp)
        print(f"  {mode} terminé ({(time.time() - t0) / 60:.1f} min)", flush=True)

    print("\n" + "=" * 130 + "\nRÉSULTATS (coût 0,15 $ / trade ; $ pour 1 once = 0,01 lot) — stop 1,5 ATR, objectif 2 ATR, 60 min max\n" + "=" * 130)
    rows = []
    for k in S:
        ii, pp = results[("REEL", k)]; mi, mp = results[("MIROIR", k)]
        tr = summarize(ii, pp, years, ndays, 2016, 2023, 0.15); va = summarize(ii, pp, years, ndays, 2024, 2024, 0.15)
        te = summarize(ii, pp, years, ndays, 2025, 2026, 0.15); te3 = summarize(ii, pp, years, ndays, 2025, 2026, 0.30)
        mi_te = summarize(mi, mp, years, ndays, 2025, 2026, 0.15)
        brut = summarize(ii, pp, years, ndays, 2025, 2026, 0.0)
        rows.append(dict(strategie=k, trades_jour=te["tpj"], reussite_test=te["reussite"], PF_test=te["PF"],
                         train_2016_23=tr["total"], val_2024=va["total"], TEST_2025_26=te["total"], test_cout_030=te3["total"],
                         test_sans_cout=brut["total"], miroir_test=mi_te["total"]))
    R = pd.DataFrame(rows).sort_values("TEST_2025_26", ascending=False)
    pd.set_option("display.width", 250)
    print(R.to_string(index=False))
    print("\nlecture : TEST = 2025-2026 jamais utilisé pour choisir ; miroir = même test sur les prix inversés (marché en baisse).")
    gagn = R[(R.train_2016_23 > 0) & (R.val_2024 > 0) & (R.TEST_2025_26 > 0) & (R.miroir_test > 0)]
    print(f"stratégies gagnantes PARTOUT (2016-23, 2024, 2025-26 et miroir) : {len(gagn)} -> {', '.join(gagn.strategie) or 'aucune'}")
    print(f"\nterminé en {(time.time() - t0) / 60:.1f} min")
