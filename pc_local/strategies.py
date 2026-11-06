# Copie de research/scalp_strategies.py (ne pas modifier ici)
# ── 21 STRATÉGIES OR M5 MULTI-TIMEFRAMES (outils donnés à l'agent RL) ─────────────────────────
# Chaque stratégie renvoie, à la clôture de chaque bougie M5 : +1 (BUY), -1 (SELL) ou 0 (rien).
# Unités supérieures lues sur bougies FERMÉES uniquement ; swings confirmés 2 bougies après (pas de fuite).
# Source des règles : liste de l'utilisateur + DeepSeek (21/10/2026), traduites en règles exactes.
import numpy as np, pandas as pd

def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def _atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def _rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def _macd(c):
    m = _ema(c, 12) - _ema(c, 26); sg = _ema(m, 9); return m, sg, m - sg

def _supertrend_dir(h, l, c, n=10, mult=3.0):
    atr = _atr(h, l, c, n).values; hl2 = ((h + l) / 2).values; cv = c.values
    up, dn = hl2 - mult * atr, hl2 + mult * atr
    fu, fd = up.copy(), dn.copy(); d = np.ones(len(cv))
    for i in range(1, len(cv)):
        fu[i] = up[i] if (up[i] > fu[i-1] or cv[i-1] < fu[i-1]) else fu[i-1]
        fd[i] = dn[i] if (dn[i] < fd[i-1] or cv[i-1] > fd[i-1]) else fd[i-1]
        d[i] = 1 if cv[i] > fd[i-1] else (-1 if cv[i] < fu[i-1] else d[i-1])
    return pd.Series(d, index=c.index)

def _heikin(o, h, l, c):
    hc = (o + h + l + c) / 4
    ho = np.empty(len(c)); ho[0] = (o.iloc[0] + c.iloc[0]) / 2; hcv = hc.values
    for i in range(1, len(c)): ho[i] = (ho[i-1] + hcv[i-1]) / 2
    return pd.Series(np.sign(hcv - ho), index=c.index)            # +1 vert, -1 rouge

def _swings(h, l, k=2):
    """Dernier swing haut / bas CONFIRMÉ (connu k bougies après la pointe)."""
    hi = h.rolling(2 * k + 1, center=True).max() == h
    lo = l.rolling(2 * k + 1, center=True).min() == l
    sh = h.where(hi).shift(k).ffill(); sl = l.where(lo).shift(k).ffill()
    return sh, sl

def _bars(m1, rule):
    return m1.resample(rule).agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna(subset=["c"])

def _to_m5(series_htf, rule, t_dec):
    """Valeur d'une unité supérieure, disponible à sa CLÔTURE, lue à l'heure de décision M5."""
    s = series_htf.copy(); s.index = s.index + pd.Timedelta(rule)
    return s.reindex(s.index.union(t_dec)).ffill().reindex(t_dec).values

def strategy_signals(m1, m5, base_min=5):
    """m1 : DataFrame o,h,l,c,v (index temps, UTC). m5 : bougies de base (M5, ou M15 avec base_min=15 ; index = ouverture).
    -> DataFrame 21 colonnes (+ 21 « _r3 »)."""
    t_dec = m5.index + pd.Timedelta(minutes=base_min)
    o, h, l, c, v = m5.o, m5.h, m5.l, m5.c, m5.v
    atr5 = _atr(h, l, c); body = (c - o).abs()
    green, red = c > o, c < o
    H1, H4, M15 = _bars(m1, "1h"), _bars(m1, "4h"), _bars(m1, "15min")
    tf = lambda s, r: pd.Series(_to_m5(s, r, t_dec), index=m5.index)
    bias1 = np.sign(tf(_ema(H1.c, 50) - _ema(H1.c, 200), "1h"))
    bias4 = np.sign(tf(_ema(H4.c, 50) - _ema(H4.c, 200), "4h"))
    S = {}
    def both(name, buy, sell): S[name] = np.where(buy.fillna(False), 1, np.where(sell.fillna(False), -1, 0))
    cross_up = lambda a, b: (a > b) & (a.shift(1) <= b.shift(1))
    cross_dn = lambda a, b: (a < b) & (a.shift(1) >= b.shift(1))

    # 1 Pullback EMA50 M5 dans le biais H1
    e50 = _ema(c, 50)
    both("s01_pullback_ema", (bias1 > 0) & (l <= e50) & (c > e50), (bias1 < 0) & (h >= e50) & (c < e50))
    # 2 Alignement MACD H4 / croisement M5
    hist4 = tf(_macd(H4.c)[2], "4h"); m, sg, _ = _macd(c)
    both("s02_macd_align", (hist4 > 0) & cross_up(m, sg), (hist4 < 0) & cross_dn(m, sg))
    # 3 Supertrend H1 + retournement Supertrend M5
    st1 = tf(_supertrend_dir(H1.h, H1.l, H1.c), "1h"); st5 = _supertrend_dir(h, l, c)
    both("s03_supertrend", (st1 > 0) & (st5 > 0) & (st5.shift(1) < 0), (st1 < 0) & (st5 < 0) & (st5.shift(1) > 0))
    # 4 Ichimoku : prix H1 vs nuage + croisement Tenkan/Kijun M5
    t9 = (H1.h.rolling(9).max() + H1.l.rolling(9).min()) / 2; k26 = (H1.h.rolling(26).max() + H1.l.rolling(26).min()) / 2
    sa = ((t9 + k26) / 2).shift(26); sb = ((H1.h.rolling(52).max() + H1.l.rolling(52).min()) / 2).shift(26)
    cloud_hi, cloud_lo = tf(pd.concat([sa, sb], axis=1).max(axis=1), "1h"), tf(pd.concat([sa, sb], axis=1).min(axis=1), "1h")
    c1 = tf(H1.c, "1h")
    tk = (h.rolling(9).max() + l.rolling(9).min()) / 2; kj = (h.rolling(26).max() + l.rolling(26).min()) / 2
    both("s04_ichimoku", (c1 > cloud_hi) & cross_up(tk, kj), (c1 < cloud_lo) & cross_dn(tk, kj))
    # 5 Heikin Ashi H1 + H4 + 1re bougie HA M5 après 3 contraires
    ha1, ha4, ha5 = tf(_heikin(H1.o, H1.h, H1.l, H1.c), "1h"), tf(_heikin(H4.o, H4.h, H4.l, H4.c), "4h"), _heikin(o, h, l, c)
    prev3 = ha5.shift(1).rolling(3).sum()
    both("s05_heikin", (ha1 > 0) & (ha4 > 0) & (ha5 > 0) & (prev3 == -3), (ha1 < 0) & (ha4 < 0) & (ha5 < 0) & (prev3 == 3))
    # 6 Ribbon EMA M5 + pente EMA50 H1 + retour sur EMA21
    e8, e21 = _ema(c, 8), _ema(c, 21); slope1 = tf(_ema(H1.c, 50).diff(3), "1h")
    both("s06_ribbon", (e8 > e21) & (e21 > e50) & (slope1 > 0) & (l <= e21) & (c > e21),
         (e8 < e21) & (e21 < e50) & (slope1 < 0) & (h >= e21) & (c < e21))
    # Chandeliers de retournement M5
    lw, uw = np.minimum(o, c) - l, h - np.maximum(o, c)
    pin_bull = (lw >= 2 * body) & ((c - l) >= 0.66 * (h - l)); pin_bear = (uw >= 2 * body) & ((h - c) >= 0.66 * (h - l))
    eng_bull = green & red.shift(1) & (c >= o.shift(1)) & (o <= c.shift(1))
    eng_bear = red & green.shift(1) & (c <= o.shift(1)) & (o >= c.shift(1))
    rev_bull, rev_bear = pin_bull | eng_bull, pin_bear | eng_bear
    # 7 Rejet sur zone (swing H1) + pin bar / englobante M5
    shH1, slH1 = _swings(H1.h, H1.l); shH1, slH1 = tf(shH1, "1h"), tf(slH1, "1h")
    both("s07_zone_reject", ((l - slH1).abs() <= 0.3 * atr5) & rev_bull, ((h - shH1).abs() <= 0.3 * atr5) & rev_bear)
    # 8 Order block H4 + cassure de structure M5 (CHoCH)
    atr4 = _atr(H4.h, H4.l, H4.c)
    imp_up = (H4.c.shift(-3) - H4.h) > 2 * atr4; imp_dn = (H4.l - H4.c.shift(-3)) > 2 * atr4
    ob_b = (H4.c < H4.o) & imp_up; ob_s = (H4.c > H4.o) & imp_dn
    obb_hi = H4.h.where(ob_b).shift(3).ffill(); obb_lo = H4.l.where(ob_b).shift(3).ffill()   # connu 3 bougies H4 après
    obs_hi = H4.h.where(ob_s).shift(3).ffill(); obs_lo = H4.l.where(ob_s).shift(3).ffill()
    obb_hi, obb_lo, obs_hi, obs_lo = (tf(x, "4h") for x in (obb_hi, obb_lo, obs_hi, obs_lo))
    sh5, sl5 = _swings(h, l)
    in_b = ((l <= obb_hi) & (h >= obb_lo)).rolling(6).max().astype(bool)
    in_s = ((h >= obs_lo) & (l <= obs_hi)).rolling(6).max().astype(bool)
    both("s08_orderblock", in_b & cross_up(c, sh5), in_s & cross_dn(c, sl5))
    # 9 Fair Value Gap H1 comblé + rejet M5
    fb_top, fb_bot = H1.l.where(H1.l > H1.h.shift(2)).ffill(), H1.h.shift(2).where(H1.l > H1.h.shift(2)).ffill()
    fs_bot, fs_top = H1.h.where(H1.h < H1.l.shift(2)).ffill(), H1.l.shift(2).where(H1.h < H1.l.shift(2)).ffill()
    fb_top, fb_bot, fs_bot, fs_top = (tf(x, "1h") for x in (fb_top, fb_bot, fs_bot, fs_top))
    both("s09_fvg", (bias1 > 0) & (l <= fb_top) & (l >= fb_bot - 0.5 * atr5) & green & (lw > body),
         (bias1 < 0) & (h >= fs_bot) & (h <= fs_top + 0.5 * atr5) & red & (uw > body))
    # 10 Break of Structure M5 dans le biais H1
    both("s10_bos", (bias1 > 0) & cross_up(c, sh5), (bias1 < 0) & cross_dn(c, sl5))
    # Range asiatique (00:00-07:00 UTC) du jour, connu à partir de 07:00
    hr = t_dec.hour; day = t_dec.normalize()
    asia = (m5.index.hour < 7)
    a_hi = h.where(asia).groupby(day).transform("max"); a_lo = l.where(asia).groupby(day).transform("min")
    after = pd.Series(hr >= 7, index=m5.index)
    run_lo = l.where(after).groupby(day).cummin(); run_hi = h.where(after).groupby(day).cummax()
    sess = pd.Series((hr >= 7) & (hr < 16), index=m5.index)
    # 11 Sweep du range asiatique puis réintégration
    both("s11_asia_sweep", sess & (run_lo < a_lo) & cross_up(c, a_lo), sess & (run_hi > a_hi) & cross_dn(c, a_hi))
    # 12 Stop hunt sur double creux / double sommet M15 (deux swings à moins de 0,1 ATR), cassé puis repris en M5
    atr15 = _atr(M15.h, M15.l, M15.c)
    def eq_level(x, is_low):
        piv = (x.rolling(5, center=True).min() == x) if is_low else (x.rolling(5, center=True).max() == x)
        conf = x.where(piv).shift(2).dropna()                      # swing confirmé 2 bougies M15 après
        prev = conf.shift(1)
        eq = (conf - prev).abs() <= 0.1 * atr15.reindex(conf.index)
        lvl = (np.minimum(conf, prev) if is_low else np.maximum(conf, prev)).where(eq)
        return lvl.reindex(M15.index).ffill()
    lvl_lo, lvl_hi = tf(eq_level(M15.l, True), "15min"), tf(eq_level(M15.h, False), "15min")
    both("s12_equal_hunt", (l < lvl_lo) & (c > lvl_lo), (h > lvl_hi) & (c < lvl_hi))
    # 13 Cassure + retest d'un swing H1
    shH1_raw, slH1_raw = _swings(H1.h, H1.l)
    brk_up, brk_dn = H1.c > shH1_raw, H1.c < slH1_raw
    lvl_b = shH1_raw.where(brk_up & ~brk_up.shift(1, fill_value=False)).ffill()
    lvl_s = slH1_raw.where(brk_dn & ~brk_dn.shift(1, fill_value=False)).ffill()
    lvl_b, lvl_s = tf(lvl_b, "1h"), tf(lvl_s, "1h")
    both("s13_break_retest", ((l - lvl_b).abs() <= 0.3 * atr5) & (c > lvl_b) & green,
         ((h - lvl_s).abs() <= 0.3 * atr5) & (c < lvl_s) & red)
    # 14 Rebond VWAP journalier dans le biais H1
    tp = (m1.h + m1.l + m1.c) / 3; vv = m1.v.replace(0, np.nan).fillna(1e-9)
    dkey = m1.index.normalize()
    vwap1 = (tp * vv).groupby(dkey).cumsum() / vv.groupby(dkey).cumsum()
    vwap = pd.Series(vwap1.reindex(t_dec - pd.Timedelta(minutes=1), method="ffill").values, index=m5.index)
    both("s14_vwap", (bias1 > 0) & (l <= vwap) & (c > vwap), (bias1 < 0) & (h >= vwap) & (c < vwap))
    # 15 Squeeze Bollinger M5 + cassure dans le biais H1
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std(); bw = 4 * sd20 / ma20
    squeeze = (bw.shift(1) <= bw.shift(1).rolling(50).min() * 1.05)
    both("s15_bb_squeeze", (bias1 > 0) & squeeze & (c > ma20 + 2 * sd20), (bias1 < 0) & squeeze & (c < ma20 - 2 * sd20))
    # 16 Divergence RSI M5 quand le RSI H1 est extrême
    rsi1 = tf(_rsi(H1.c), "1h"); r5 = _rsi(c)
    lo_pt = l.rolling(5, center=True).min() == l; hi_pt = h.rolling(5, center=True).max() == h
    sl_p = l.where(lo_pt).shift(2).ffill(); sl_r = r5.where(lo_pt).shift(2).ffill()
    sh_p = h.where(hi_pt).shift(2).ffill(); sh_r = r5.where(hi_pt).shift(2).ffill()
    new_lo = sl_p != sl_p.shift(1); new_hi = sh_p != sh_p.shift(1)
    pl_p, pl_r = sl_p.shift(1).where(new_lo).ffill(), sl_r.shift(1).where(new_lo).ffill()
    ph_p, ph_r = sh_p.shift(1).where(new_hi).ffill(), sh_r.shift(1).where(new_hi).ffill()
    both("s16_rsi_div", (rsi1 < 35) & new_lo & (sl_p < pl_p) & (sl_r > pl_r), (rsi1 > 65) & new_hi & (sh_p > ph_p) & (sh_r < ph_r))
    # 17 Points pivots journaliers (veille) + rejet M5, biais H4
    D = _bars(m1, "1D"); P = (D.h + D.l + D.c) / 3; S1 = 2 * P - D.h; R1 = 2 * P - D.l
    P, S1, R1 = tf(P, "1D"), tf(S1, "1D"), tf(R1, "1D")
    near_sup = ((l <= P) & (c > P)) | ((l <= S1) & (c > S1)); near_res = ((h >= P) & (c < P)) | ((h >= R1) & (c < R1))
    both("s17_pivots", (bias4 > 0) & near_sup & (lw > body), (bias4 < 0) & near_res & (uw > body))
    # 18 Retracement Fibonacci 50-61,8 % du dernier mouvement H1
    sw_lo, sw_hi = tf(_swings(H1.h, H1.l)[1], "1h"), tf(_swings(H1.h, H1.l)[0], "1h")
    rng = sw_hi - sw_lo
    f50_b, f618_b = sw_hi - 0.5 * rng, sw_hi - 0.618 * rng
    f50_s, f618_s = sw_lo + 0.5 * rng, sw_lo + 0.618 * rng
    both("s18_fibo", (bias1 > 0) & (l <= f50_b) & (l >= f618_b) & rev_bull, (bias1 < 0) & (h >= f50_s) & (h <= f618_s) & rev_bear)
    # 19 Breakout ATR : volatilité H4 en hausse + grosse bougie M5 dans le biais
    atr4_up = tf(atr4.diff(), "4h") > 0
    big = (h - l) > 2 * atr5
    both("s19_atr_breakout", atr4_up & big & green & (bias1 > 0), atr4_up & big & red & (bias1 < 0))
    # 20 Breakout de Londres (07-10 UTC) si volatilité H1 suffisante
    atr1 = _atr(H1.h, H1.l, H1.c); vol_ok = tf(atr1 > atr1.rolling(500).median(), "1h") > 0
    lon = pd.Series((hr >= 7) & (hr < 10), index=m5.index)
    both("s20_london_breakout", lon & vol_ok & cross_up(c, a_hi), lon & vol_ok & cross_dn(c, a_lo))
    # 21 Killzone New York (12-14 UTC) : biais de Londres + cassure M5 avec volume
    lo_open = o.where(pd.Series(m5.index.hour == 7, index=m5.index) & (m5.index.minute == 0)).groupby(day).transform("first")
    lo_close = c.where(pd.Series(m5.index.hour == 11, index=m5.index) & (m5.index.minute == 60 - base_min)).groupby(day).transform("first")
    ldn_bias = np.sign(lo_close - lo_open)
    ny = pd.Series((hr >= 12) & (hr < 14), index=m5.index)
    vol_hi = v > 1.2 * v.rolling(20).mean()
    both("s21_ny_killzone", ny & (ldn_bias > 0) & (c > h.shift(1).rolling(6).max()) & vol_hi,
         ny & (ldn_bias < 0) & (c < l.shift(1).rolling(6).min()) & vol_hi)

    out = pd.DataFrame(S, index=m5.index).astype(np.float32)
    recent = out.rolling(3, min_periods=1).sum().clip(-1, 1)          # signal donné dans les 3 dernières bougies
    recent.columns = [f"{k}_r3" for k in out.columns]
    return pd.concat([out, recent], axis=1)
