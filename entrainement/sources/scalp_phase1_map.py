"""SCALPING M5 — Phase 1 : carte de prévisibilité (où l'or M5 est-il prévisible, et jusqu'à quel coût ?).

Données : Dukascopy XAUUSD M1 bid/ask (sortie de la phase 0), prix mid, UTC.
Décision à la clôture de chaque bougie M5. Features multi-timeframes SANS fuite :
  - M5 : rendements, range, corps, mèches, ATR/prix, RSI, distance EMA20/50 (en ATR), MACD hist (en ATR),
    position Bollinger, ADX/DI, fenêtre des 12 dernières bougies (rendements/range en ATR) ;
  - M15 / H1 / H4 / D1 : mêmes indicateurs résumés, sur bougies FERMÉES uniquement (alignées sur leur clôture) ;
  - spread du moment, heure / jour / session.
Cibles : signe du mouvement du prix mid à 5, 10, 15, 30, 60 minutes.
Modèle : LightGBM, walk-forward annuel (entraînement 2016 -> Y-1, test Y, Y = 2018..2026), purge 1 jour.
Carte : pour chaque horizon x session x volatilité : AUC, et « coût maximal tolérable » =
  gain moyen (en $ par once) des 20 % de signaux les plus confiants, pris dans le sens prédit.
"""
import os, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score

INPUT = Path("/kaggle/input"); OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
HORIZONS = [5, 10, 15, 30, 60]
YEARS = list(range(2018, 2027))
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

if __name__ == "__main__":
    t0 = time.time()
    m = pd.read_parquet(next(INPUT.rglob("xau_m1_bidask.parquet")))
    m = m.set_index("datetime").sort_index()
    m1 = pd.DataFrame({"o": (m.o_bid + m.o_ask) / 2, "h": (m.h_bid + m.h_ask) / 2, "l": (m.l_bid + m.l_ask) / 2,
                       "c": (m.c_bid + m.c_ask) / 2, "spread": m.spread})
    m5 = ohlc(m1, "5min")
    F, atr5 = ind(m5, "m5")
    lags = {}
    for k in range(1, 12):
        lags[f"m5_ret_l{k}"] = (F["m5_ret1"] / F["m5_atr_n"]).shift(k)
        lags[f"m5_range_l{k}"] = F["m5_range"].shift(k)
    F = pd.concat([F, pd.DataFrame(lags, index=F.index)], axis=1)
    F["spread"] = m5.spread
    t_dec = m5.index + pd.Timedelta(minutes=5)                       # heure de décision = clôture M5
    for name, rule in TF.items():
        b = ohlc(m1, rule)
        f, _ = ind(b, name.lower())
        f.index = f.index + pd.Timedelta(rule)                       # disponible à la CLÔTURE seulement
        F = pd.merge_asof(F.assign(_t=t_dec).sort_values("_t"), f.sort_index(), left_on="_t", right_index=True,
                          direction="backward").drop(columns="_t").set_index(m5.index)
    hr = t_dec.hour + t_dec.minute / 60
    F["h_sin"], F["h_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    F["dow"] = t_dec.dayofweek
    session = np.select([(hr >= 0) & (hr < 7), (hr >= 7) & (hr < 12), (hr >= 12) & (hr < 16), (hr >= 16) & (hr < 21)],
                        ["Asie", "Londres", "Londres+NY", "NY"], "Nuit")
    F = F.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    print(f"M5 : {len(F):,} bougies x {F.shape[1]} features ({(time.time()-t0)/60:.1f} min)", flush=True)

    mid = m1.c
    tgt = {}
    for h in HORIZONS:
        now = mid.reindex(t_dec, method="ffill").values
        fut_t = t_dec + pd.Timedelta(minutes=h)
        fut = mid.reindex(fut_t, method="ffill", tolerance=pd.Timedelta(minutes=2)).values
        tgt[h] = fut - now
    years = t_dec.year.values
    voltile = pd.Series(F["m5_atr_z"].values).rank(pct=True).values      # tercile de volatilité (relatif)
    voltile = np.where(voltile < 1 / 3, "calme", np.where(voltile < 2 / 3, "normal", "agité"))

    params = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.5,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=os.cpu_count())
    X = F.values
    preds = {h: np.full(len(F), np.nan) for h in HORIZONS}
    for h in HORIZONS:
        y = tgt[h]
        for Y in YEARS:
            tr = (years < Y - 0) & (t_dec < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=1)) & ~np.isnan(y)
            te = (years == Y)
            tri = np.where(tr)[0]; cut = int(len(tri) * 0.9)
            fit, va = tri[:cut], tri[cut:]
            mdl = lgb.train(params, lgb.Dataset(X[fit], (y[fit] > 0).astype(int)), 2000,
                            valid_sets=[lgb.Dataset(X[va], (y[va] > 0).astype(int))],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
            preds[h][te] = mdl.predict(X[te], num_iteration=mdl.best_iteration)
        print(f"  horizon {h} min : OK ({(time.time()-t0)/60:.1f} min)", flush=True)

    def cell(mask, h):
        y, p = tgt[h][mask], preds[h][mask]
        ok = ~np.isnan(y) & ~np.isnan(p)
        y, p = y[ok], p[ok]
        if len(y) < 2000: return None
        auc = roc_auc_score(y > 0, p)
        conf = np.abs(p - 0.5); top = conf >= np.quantile(conf, 0.8)          # 20 % les plus confiants
        gain = (np.sign(p[top] - 0.5) * y[top]).mean()                        # $ par once, avant coûts
        return auc, gain, top.sum()

    rows = []
    for h in HORIZONS:
        for sess in ["TOUT", "Asie", "Londres", "Londres+NY", "NY", "Nuit"]:
            for vol in ["TOUT", "calme", "normal", "agité"]:
                base_mask = (years >= 2018)
                if sess != "TOUT": base_mask &= (session == sess)
                if vol != "TOUT": base_mask &= (voltile == vol)
                r = cell(base_mask, h)
                if r: rows.append(dict(horizon=h, session=sess, volatilite=vol, AUC=round(r[0], 4),
                                       cout_max_tolerable=round(r[1], 3), n_signaux=int(r[2])))
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 100 + "\nCARTE — toutes sessions / volatilités, 2018-2026 hors échantillon\n" + "=" * 100)
    print(tab[(tab.session == "TOUT") & (tab.volatilite == "TOUT")].to_string(index=False))
    print("\nPAR SESSION (toutes volatilités) :")
    print(tab[(tab.volatilite == "TOUT")].pivot(index="session", columns="horizon", values="cout_max_tolerable").to_string())
    print("\nPAR SESSION — AUC :")
    print(tab[(tab.volatilite == "TOUT")].pivot(index="session", columns="horizon", values="AUC").to_string())
    print("\nPAR VOLATILITÉ (toutes sessions) — coût max tolérable :")
    print(tab[(tab.session == "TOUT")].pivot(index="volatilite", columns="horizon", values="cout_max_tolerable").to_string())
    print("\nMEILLEURES CASES (coût max tolérable le plus élevé) :")
    print(tab.sort_values("cout_max_tolerable", ascending=False).head(15).to_string(index=False))

    print("\nSTABILITÉ PAR ANNÉE (toutes sessions) — coût max tolérable :")
    yr = []
    for h in HORIZONS:
        for Y in YEARS:
            r = cell(years == Y, h)
            if r: yr.append(dict(annee=Y, horizon=h, cout=round(r[1], 3), AUC=round(r[0], 4)))
    yr = pd.DataFrame(yr)
    print(yr.pivot(index="annee", columns="horizon", values="cout").to_string())
    print(yr.pivot(index="annee", columns="horizon", values="AUC").to_string())
    tab.to_csv(OUT / "scalp_map.csv", index=False); yr.to_csv(OUT / "scalp_map_years.csv", index=False)
    pd.DataFrame({h: preds[h] for h in HORIZONS}, index=F.index).astype(np.float32).to_parquet(OUT / "scalp_preds_wf.parquet")
    print(f"\nTerminé en {(time.time()-t0)/60:.1f} min")
