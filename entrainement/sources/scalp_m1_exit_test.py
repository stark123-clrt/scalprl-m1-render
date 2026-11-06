
# ════════════ COMITÉ OR M1 (agents 1, 2, 3, 5, 6) : TEST D'UNE SORTIE D'URGENCE PLUS RAPIDE ════════════
# Demande de l'utilisateur (05/11/2026, après un SL de sécurité 25 $ touché en 6 min pendant une chute) :
# la règle de tendance (EMA50 M15) ne réagit qu'à la FERMETURE d'une bougie M15 -> une chute rapide passe à travers.
# Chargé après phase 1, agent v4, agent v5 (GS_STRICT=1, règle m15_d_ema50), stratégies M1 et agent M1.
# Simulation minute par minute avec les prix bid/ask Dukascopy (stops touchés DANS la minute, sauts de prix au prix d'ouverture).
# Variantes (une à la fois, puis combinaisons) :
#   A  comité seul (comme le test 2025-2026)                 B  + SL de sécurité 25 $ (= robot en démo)
#   C  + sortie d'urgence à k x ATR M1 sous le prix d'entrée (k = 2, 3, 4, 6), SL 25 $ gardé
#   D  + « tendance rapide » : à chaque minute, on recalcule l'écart à l'EMA50 M15 avec la bougie M15 EN COURS ;
#        si la position va contre (écart < -0,3 ATR M15) -> sortie (et éventuellement pas d'entrée contre)
#   attente : après une sortie d'urgence, pas de nouvelle entrée dans le même sens pendant 15 min.
# Périodes : VAL 2024 et TEST 2025-2026, monde réel et monde miroir (marché en baisse). Rien n'est modifié en direct.
import re, shutil
COMMITTEE = [int(s) for s in os.environ.get("GS_COMMITTEE", "1,2,3,5,6").split(",")]
COOL = 15                                                                   # minutes d'attente après une sortie d'urgence

def agent_zip(s):
    for p in sorted(INPUT.rglob(f"ppo_m1_seed{s}_best*")):
        z = OUT / f"ppo_m1_seed{s}_best.zip"
        if p.is_dir(): shutil.make_archive(str(z)[:-4], "zip", p); return z
        if p.suffix == ".zip": shutil.copy(p, z); return z
    raise FileNotFoundError(f"agent {s} introuvable dans {INPUT}")

class ExitEnv(ScalpEnv):
    """Comité M1 + règle stricte (EMA50 M15 fermée) + stops dans la minute suivante (bid pour un achat, ask pour une vente)."""
    def __init__(self, X, price, atr, brk, day, lo, hi, side, fastd, jn, sprr, sl=None, k=None, fast=False, fast_entry=False,
                 cool=False, mv=None, knife=None, spr=None):
        super().__init__([X], [price], atr, brk, day, lo, hi, 0.0, random_start=False)
        self.side, self.fastd, self.jn, self.sprr = side, fastd, jn, sprr   # side = (O achat, pire achat, O vente, pire vente)
        self.sl, self.k, self.fast, self.fast_entry, self.cool = sl, k, fast, fast_entry, cool
        self.mv, self.knife, self.spr = mv, knife, spr
    def reset(self, seed=None, options=None):
        o, i = super().reset(seed=seed)
        self.reasons, self.cool_dir, self.cool_until, self.stop, self.kind = [], 0, -1, None, ""
        return o, i
    def _close_at(self, px, reason):
        self.trades.append(self.pos * (px - self.entry)); self.sides.append(self.pos); self.holds.append(self.age)
        self.reasons.append(reason); self.pos, self.age, self.stop = 0, 0, None
    def step(self, a):
        t = self.t; sgn = 1.0 if self.p[self.lo] > 0 else -1.0
        al = sgn * ALLOWED[t]; fd = sgn * self.fastd[t]
        tgt = (0, 1, -1)[int(a)]
        if self.pos != 0 and al == -self.pos: tgt = 0                               # règle stricte (bougie M15 fermée)
        if tgt != 0 and tgt != self.pos and tgt != al: tgt = 0
        mv3 = self.p[t] - self.p[max(t - 3, 0)]                                     # mouvement des 3 dernières minutes
        spr_bad = self.spr is not None and self.sprr[t] > self.spr
        forced = ""
        if self.pos != 0:
            if self.fast and self.pos * fd < -0.3: forced = "tendance rapide"       # la bougie M15 en cours a tourné
            elif self.mv and self.pos * mv3 < -self.mv * self.atr[t]: forced = "mouvement anormal"
            elif spr_bad: forced = "spread anormal"
        if forced: tgt = 0
        if tgt != 0 and tgt != self.pos:
            if self.fast_entry and tgt * fd <= 0.3: tgt = 0
            elif self.knife and tgt * mv3 < -self.knife * self.atr[t]: tgt = 0        # couteau qui tombe : pas d'entrée
            elif spr_bad: tgt = 0
            elif self.cool and tgt == self.cool_dir and t < self.cool_until: tgt = 0
        if self.brk[t] or self.age >= MAX_HOLD: tgt = 0
        if tgt != self.pos:
            if self.pos != 0:
                if forced: self.cool_dir, self.cool_until = self.pos, t + COOL
                self._close_at(self.p[t], forced or "agent")
            if tgt != 0:
                self.pos, self.entry, self.age, self.n_day = tgt, self.p[t], 0, self.n_day + 1
                cand = [(self.sl, "SL 25$")] if self.sl else []
                if self.k: cand.append((self.k * self.atr[t], "urgence"))
                if cand:
                    d, self.kind = min(cand); self.stop = self.entry - tgt * d
        if self.pos != 0 and self.stop is not None and not self.brk[t]:              # minute suivante : stop touché ?
            j = self.jn[t]
            O, W = (self.side[0][j], self.side[1][j]) if self.pos == 1 else (self.side[2][j], self.side[3][j])
            if self.pos * (W - self.stop) <= 0:
                if self.kind == "urgence": self.cool_dir, self.cool_until = self.pos, t + 1 + COOL
                self._close_at(O if self.pos * (O - self.stop) < 0 else self.stop, self.kind)
        nxt = t + 1
        if self.pos != 0: self.age += 1
        if self.day[nxt] != self.day[t]: self.n_day = 0
        self.t = nxt
        self.upnl = self.pos * (self.p[self.t] - self.entry) if self.pos else 0.0
        done = self.t >= self.t_end
        if done and self.pos != 0: self._close_at(self.p[self.t], "fin")
        return self._obs(), 0.0, done, False, {}

def run(com, world, lo, hi, rule):
    X_, p_, side = world
    env = ExitEnv(X_, p_, atr_a, brk, day, lo, hi, side, fastd, jn, sprr, **rule)
    obs, _ = env.reset(); done = False
    while not done:
        obs, _, done, _, _ = env.step(com.predict(obs)[0])
    g = np.array(env.trades) - 0.15; rs = np.array(env.reasons); eq = np.cumsum(g)
    days = max(len(np.unique(day[lo:hi])), 1); gl = -g[g < 0].sum()
    urg = {r: int((rs == r).sum()) for r in ("urgence", "tendance rapide", "mouvement anormal", "spread anormal", "SL 25$")
           if (rs == r).sum()}
    return dict(tot=g.sum(), txt=(f"{len(g)/days:4.1f} tr/j | 0,15$ {g.sum():+6.0f} | PF {g[g > 0].sum() / max(gl, 1e-9):.3f} | "
                                   f"réussite {np.mean(g > 0):.0%} | gain moy {g[g > 0].mean():.2f} perte moy {g[g <= 0].mean():.2f} | "
                                   f"pire {g.min():.1f} | pertes < -10$ : {int((g < -10).sum())} | DD {(np.maximum.accumulate(eq) - eq).max():.0f}"
                                   + (f" | sorties {urg}" if urg else "")))

if __name__ == "__main__":
    t0 = time.time()
    F, price, atr_a, gap, t_dec = load_m1_agent()
    idx = lambda a, b: (int(np.searchsorted(t_dec, pd.Timestamp(a))), int(np.searchsorted(t_dec, pd.Timestamp(b))))
    tr_lo, tr_hi = idx(*TRAIN); va_lo, va_hi = idx(*VAL); te_lo, te_hi = idx(*TEST)
    Fm = mirror_features(F)
    sub = np.arange(tr_lo, tr_hi, 7)                                    # même normalisation qu'à l'entraînement
    both = np.vstack([F.values[sub], Fm.values[sub]])
    mu, sd = np.nanmean(both, 0), np.nanstd(both, 0) + 1e-9; del both
    std = lambda A: np.clip(np.nan_to_num((A - mu) / sd), -10, 10).astype(np.float32)
    X, Xm = std(F.values), std(Fm.values); del Fm
    ts = np.nan_to_num(F[TREND_COLS[0]].values); ts2 = np.nan_to_num(F[TREND_COLS[1]].values)
    ALLOWED = np.where((np.abs(ts) > 0.3) & (np.sign(ts) == np.sign(ts2)), np.sign(ts), 0.0)
    day = t_dec.normalize().values.astype("datetime64[D]").astype(np.int64)
    YEARS_ARR = np.asarray(t_dec.year); brk = gap.copy()

    m = pd.read_parquet(next(INPUT.rglob("xau_m1_bidask.parquet"))).set_index("datetime").sort_index()
    m = m[~m.index.duplicated()]
    # « tendance rapide » : écart prix / EMA50 M15 en comptant la bougie M15 EN COURS (prix de la minute = sa clôture provisoire)
    mid = pd.DataFrame({"o": (m.o_bid + m.o_ask) / 2, "h": (m.h_bid + m.h_ask) / 2, "l": (m.l_bid + m.l_ask) / 2,
                        "c": (m.c_bid + m.c_ask) / 2, "spread": m.spread})
    b15 = ohlc(mid[mid.index >= pd.Timestamp(START_M1) - pd.Timedelta(days=30)], "15min")
    _, atr15 = ind(b15, "m15"); ema15 = b15.c.ewm(span=50, adjust=False).mean()
    end15 = (b15.index + pd.Timedelta("15min")).values
    k15 = np.searchsorted(end15, t_dec.values, side="right") - 1                 # dernière M15 fermée à la décision
    E, A = ema15.values[np.clip(k15, 0, None)], atr15.values[np.clip(k15, 0, None)]
    alpha = 2 / 51
    prov = np.where(end15[np.clip(k15, 0, None)] == t_dec.values, E, E + alpha * (price - E))
    fastd = np.nan_to_num((price - prov) / A)
    chk = np.mean(np.sign(np.where(np.abs(fastd) > 0.3, fastd, 0)) == ALLOWED)
    print(f"tendance rapide : même sens que la règle M15 fermée {chk:.1%} des minutes", flush=True)
    del mid, b15
    # prix bid/ask « équivalent milieu » (demi-spread médian du jour) : achat -> bid, vente -> ask (pics de spread compris)
    hs = ((m.c_ask - m.c_bid).groupby(m.index.normalize()).transform("median") / 2).values
    oB, lB, hB = m.o_bid.values + hs, m.l_bid.values + hs, m.h_bid.values + hs
    oA, hA, lA = m.o_ask.values - hs, m.h_ask.values - hs, m.l_ask.values - hs
    pos_m = np.searchsorted(m.index.values, F.index.values)
    assert np.all(m.index.values[pos_m] == F.index.values)
    jn = np.r_[pos_m[1:], pos_m[-1]]                                            # bougie M1 suivante (minute après la décision)
    # spread anormal : spread à la clôture de la minute / médiane des 24 h précédentes
    sp = pd.Series((m.c_ask - m.c_bid).values, index=m.index)
    sp = sp[sp.index >= pd.Timestamp(START_M1) - pd.Timedelta(days=5)]
    sprr = np.nan_to_num((sp / sp.rolling(1440, min_periods=100).median().shift(1)).reindex(F.index).values, nan=1.0)
    q = np.quantile(sprr[te_lo:te_hi], [0.5, 0.99, 0.999])
    print(f"spread / médiane 24 h (2025-2026) : médiane {q[0]:.2f}, 99 % {q[1]:.2f}, 99,9 % {q[2]:.2f} | "
          f"> 2x : {np.mean(sprr[te_lo:te_hi] > 2):.2%} des minutes, > 3x : {np.mean(sprr[te_lo:te_hi] > 3):.2%}", flush=True)
    mv3_all = np.abs(price - np.r_[price[:3], price[:-3]]) / np.maximum(atr_a, 1e-6)
    q = np.quantile(mv3_all[te_lo:te_hi], [0.5, 0.95, 0.99])
    print(f"mouvement sur 3 min en ATR M1 (2025-2026) : médiane {q[0]:.2f}, 95 % {q[1]:.2f}, 99 % {q[2]:.2f}", flush=True)
    del sp, mv3_all
    real = (X, price, (oB, lB, oA, hA))
    mirror = (Xm, -price, (-oA, -hA, -oB, -lB))                                 # miroir : achat = vente réelle (côté ask), etc.
    del m

    models = [PPO.load(agent_zip(s), device="cpu") for s in COMMITTEE]
    com = FastCommittee(models)
    a_t = atr_a[te_lo:te_hi]
    print(f"Comité OR M1 {COMMITTEE} | ATR M1 médian 2025-2026 {np.nanmedian(a_t):.2f} $ (2 ATR ~ {2*np.nanmedian(a_t):.1f} $, "
          f"4 ATR ~ {4*np.nanmedian(a_t):.1f} $) | coût 0,15 $ par trade ({(time.time()-t0)/60:.1f} min)\n", flush=True)
    RULES = [("A  comité seul (test 2025-2026)", dict()),
             ("B  + SL sécurité 25 $ (robot en démo)", dict(sl=25)),
             ("C2 + sortie d'urgence 2 ATR M1", dict(sl=25, k=2)),
             ("C3 + sortie d'urgence 3 ATR M1", dict(sl=25, k=3)),
             ("C4 + sortie d'urgence 4 ATR M1", dict(sl=25, k=4)),
             ("C6 + sortie d'urgence 6 ATR M1", dict(sl=25, k=6)),
             ("C3a + urgence 3 ATR + attente 15 min", dict(sl=25, k=3, cool=True)),
             ("C4a + urgence 4 ATR + attente 15 min", dict(sl=25, k=4, cool=True)),
             ("D  + tendance rapide (sortie)", dict(sl=25, fast=True, cool=True)),
             ("D+ + tendance rapide (sortie ET entrée)", dict(sl=25, fast=True, fast_entry=True, cool=True)),
             ("E  + tendance rapide + urgence 4 ATR", dict(sl=25, k=4, fast=True, cool=True)),
             ("F3 + mouvement anormal (3 ATR en 3 min contre)", dict(sl=25, mv=3, cool=True)),
             ("F4 + mouvement anormal (4 ATR en 3 min contre)", dict(sl=25, mv=4, cool=True)),
             ("K3 + pas d'entrée si chute > 3 ATR en 3 min", dict(sl=25, knife=3)),
             ("K4 + pas d'entrée si chute > 4 ATR en 3 min", dict(sl=25, knife=4)),
             ("FK + mouvement anormal 3 ATR + couteau 3 ATR", dict(sl=25, mv=3, knife=3, cool=True)),
             ("S2 + spread > 2x normal : sortie, pas d'entrée", dict(sl=25, spr=2.0)),
             ("S3 + spread > 3x normal : sortie, pas d'entrée", dict(sl=25, spr=3.0))]
    part = os.environ.get("GS_PART")
    if part: RULES = RULES[int(part)::2]
    summary = []
    for name, rule in RULES:
        print(f"  {name}", flush=True)
        row = []
        for per, lo, hi in (("VAL 2024 ", va_lo, va_hi), ("TEST 25-26", te_lo, te_hi)):
            for wname, w in (("réel  ", real), ("miroir", mirror)):
                r = run(com, w, lo, hi, rule); row.append(r["tot"])
                print(f"    {per} {wname} : {r['txt']}", flush=True)
        summary.append((name, row))
        print(f"    ({(time.time()-t0)/60:.0f} min)", flush=True)
    print("\n" + "=" * 100 + "\nRÉSUMÉ à 0,15 $ par trade : VAL 2024 réel / miroir | TEST 2025-2026 réel / miroir | total TEST\n" + "=" * 100)
    for name, (vr, vm, tr_, tm) in summary:
        print(f"  {name:42s} {vr:+7.0f} / {vm:+7.0f} | {tr_:+7.0f} / {tm:+7.0f} | {tr_ + tm:+7.0f}")
    print(f"\nTerminé en {(time.time()-t0)/60:.1f} min")
