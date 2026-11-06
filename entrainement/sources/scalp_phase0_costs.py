"""SCALPING M5 — Phase 0 : mesurer la réalité (coûts) avant de construire quoi que ce soit.

1. Télécharge l'historique Dukascopy XAUUSD en bougies M1 BID et ASK (2016 -> aujourd'hui, UTC).
2. Spread réel = ask - bid, minute par minute : par année, par heure UTC, par jour de la semaine.
3. Mouvement typique du prix sur 5 / 15 / 30 / 60 minutes, comparé au spread : le rapport coût / mouvement dit
   où le scalping est physiquement possible.
4. Compare le prix Dukascopy au prix Deriv (source de l'API) sur la période commune.
Sortie : /kaggle/working/xau_m1_bidask.parquet (réutilisé par les phases suivantes) + tableaux.
"""
import io, lzma, os, time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import numpy as np, pandas as pd, requests

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
START, END = date(2016, 1, 1), date.today()
URL = "https://datafeed.dukascopy.com/datafeed/XAUUSD/{y}/{m:02d}/{d:02d}/{side}_candles_min_1.bi5"
DT = np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"), ("h", ">i4"), ("v", ">f4")])
sess = requests.Session(); sess.headers["User-Agent"] = "Mozilla/5.0"

def fetch(day, side):
    url = URL.format(y=day.year, m=day.month - 1, d=day.day, side=side)    # mois indexés à partir de 0
    for attempt in range(4):
        try:
            r = sess.get(url, timeout=30)
            if r.status_code == 404 or not r.content: return None
            r.raise_for_status()
            a = np.frombuffer(lzma.decompress(r.content), dtype=DT)
            if len(a) == 0: return None
            base = pd.Timestamp(day)
            return pd.DataFrame({"datetime": base + pd.to_timedelta(a["t"].astype(np.int64), unit="s"),
                                 "o": a["o"], "h": a["h"], "l": a["l"], "c": a["c"], "v": a["v"]})
        except Exception:
            time.sleep(1 + 2 * attempt)
    return None

def download(side):
    days = [START + timedelta(n) for n in range((END - START).days + 1)]
    days = [d for d in days if d.weekday() != 5]                            # samedi : marché fermé
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(lambda d: fetch(d, side), days) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    scale = 1000.0 if df.c.median() > 100000 else (100.0 if df.c.median() > 10000 else 1.0)
    for k in "ohlc": df[k] = (df[k] / scale).astype(np.float64)
    print(f"  {side} : {len(df):,} bougies M1, {df.datetime.min()} -> {df.datetime.max()} (échelle /{scale:g})", flush=True)
    return df

if __name__ == "__main__":
    t0 = time.time()
    print("[1] Téléchargement Dukascopy XAUUSD M1 BID + ASK ...", flush=True)
    bid, ask = download("BID"), download("ASK")
    m = bid.merge(ask[["datetime", "o", "h", "l", "c"]], on="datetime", suffixes=("_bid", "_ask"))
    m = m[(m.v > 0)]                                                        # minutes sans activité retirées
    m["spread"] = m.c_ask - m.c_bid
    m["mid"] = (m.c_ask + m.c_bid) / 2
    m.to_parquet(OUT / "xau_m1_bidask.parquet")
    print(f"  fusion : {len(m):,} minutes ({(time.time()-t0)/60:.1f} min)\n", flush=True)

    m["year"] = m.datetime.dt.year; m["hour"] = m.datetime.dt.hour; m["dow"] = m.datetime.dt.dayofweek
    print("[2] SPREAD réel (ask - bid, en $) par année :")
    print(m.groupby("year").spread.describe(percentiles=[.5, .9, .99])[["count", "mean", "50%", "90%", "99%"]].round(3).to_string())
    print("\n    par heure UTC (toutes années) :")
    print(m.groupby("hour").spread.agg(["median", lambda s: s.quantile(.9)]).rename(columns={"<lambda_0>": "p90"}).round(3).T.to_string())
    print("\n    par heure UTC, 2026 seulement :")
    print(m[m.year == 2026].groupby("hour").spread.median().round(3).to_frame().T.to_string())

    print("\n[3] MOUVEMENT du prix (mid) vs SPREAD — le rapport qui décide si le scalping est possible")
    s = m.set_index("datetime").mid
    rows = []
    for mins in (5, 15, 30, 60):
        fut = s.shift(-mins, freq="min").reindex(s.index)
        mv = (fut - s).abs()
        d = pd.DataFrame({"mv": mv, "spread": m.set_index("datetime").spread, "year": s.index.year}).dropna()
        g = d.groupby("year").agg(mouvement_median=("mv", "median"), spread_median=("spread", "median"))
        g["cout_en_%_du_mouvement"] = (100 * g.spread_median / g.mouvement_median).round(1)
        g["horizon_min"] = mins
        rows.append(g.reset_index())
    tab = pd.concat(rows).pivot(index="year", columns="horizon_min", values="cout_en_%_du_mouvement")
    print("    coût (spread médian) en % du mouvement médian absolu, par horizon (minutes) :")
    print(tab.to_string())
    mv5 = pd.concat(rows); print("\n    mouvement médian (en $) :")
    print(mv5.pivot(index="year", columns="horizon_min", values="mouvement_median").round(2).to_string())

    der = sorted(Path("/kaggle/input").rglob("DAT_DERIV_XAUUSD_M1_*.csv")) if Path("/kaggle/input").exists() else []
    if der:
        print("\n[4] Dukascopy vs Deriv (période commune) :")
        d = pd.concat([pd.read_csv(p, sep=";", header=None, names=["dt", "o", "h", "l", "c", "v"]) for p in der])
        d["datetime"] = pd.to_datetime(d.dt, format="%Y%m%d %H%M%S") + pd.Timedelta(hours=5)
        j = m.merge(d[["datetime", "c"]].rename(columns={"c": "deriv"}), on="datetime")
        diff = j.deriv - j.mid
        print(f"    {len(j):,} minutes communes | écart Deriv - Dukascopy mid : médiane {diff.median():+.3f} $, "
              f"écart absolu médian {diff.abs().median():.3f} $ | corrélation des rendements M1 "
              f"{np.corrcoef(j.deriv.diff().fillna(0), j.mid.diff().fillna(0))[0,1]:.4f}")
    print(f"\nTerminé en {(time.time()-t0)/60:.1f} min")
