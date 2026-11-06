
# ════════════ AGENT RL M5 v5 : SUIVI DE TENDANCE ════════════
# Chargé après phase 1, stratégies et l'agent v4 (load, ScalpEnv, evaluate, mirror_features...). Différences avec la v4 :
#   1. GoldStark retiré (pas de source wf_probs) ;
#   2. semaines d'entraînement tirées plus souvent quand le marché TENDANCE (hausse ou baisse), en plus des années récentes ;
#   3. petit bonus de récompense par bougie quand la position va dans le sens de la tendance H1 nette
#      (distance du prix H1 à son EMA50 > 0,5 ATR), malus si elle va contre (GS_TREND_BONUS, en ATR par bougie) ;
#   4. choix du meilleur entraînement sur 2024 RÉEL + 2024 MIROIR (= une année de baisse) ;
#   5. test final 2025-2026 réel + miroir (baisse) + placebo propre (sens au hasard, l'agent ne gère que la sortie).
TREND_BONUS = float(os.environ.get("GS_TREND_BONUS", 0.005))
TREND_SAMPLING = float(os.environ.get("GS_TREND_SAMPLING", 1.0))      # poids supplémentaire des semaines en tendance
STRICT = os.environ.get("GS_STRICT", "0") == "1"                       # v5 STRICTE : entrées seulement dans le sens H1 + M15
TREND_COLS = os.environ.get("GS_TREND_COLS", "h1_d_ema50,m15_d_ema50").split(",")   # règle : 2 écarts prix / EMA (en ATR)
INIT_FROM = os.environ.get("GS_INIT_FROM", "")                  # reprendre un agent existant (ex. « v5s ») au lieu de partir de zéro
TAG = os.environ.get("GS_TAG", "v5s" if STRICT else "v5")
ALLOWED = None                                                         # sens autorisé par bougie (+1 / -1 / 0), monde réel

class TrendEnv(ScalpEnv):
    def __init__(self, Xs, prices, trends, cw, *a, **k):
        super().__init__(Xs, prices, *a, **k)
        self.trends, self.cw = trends, cw
    def reset(self, seed=None, options=None):
        o, i = super().reset(seed=seed, options=options)
        self.tr = self.trends[[id(x) for x in self.Xs].index(id(self.X))]
        return o, i
    def step(self, a):
        t0, held = self.t, (0, 1, -1)[int(a)]
        if self.brk[t0] or self.age >= MAX_HOLD: held = 0
        obs, r, done, trunc, info = super().step(a)
        return obs, r + TREND_BONUS * held * self.tr[t0], done, trunc, info

class _Strict:
    """Règle de suivi de tendance : entrée seulement dans le sens où H1 et M15 sont alignés ; une position contraire à la
    tendance est fermée. Dans le monde miroir (prix négatifs), le sens autorisé est inversé."""
    def step(self, a):
        al = (ALLOWED if self.p[self.lo] > 0 else -ALLOWED)[self.t]
        tgt = (0, 1, -1)[int(a)]
        if self.pos != 0 and al == -self.pos: tgt = 0
        if tgt != 0 and tgt != self.pos and tgt != al: tgt = 0
        return super().step({0: 0, 1: 1, -1: 2}[tgt])

class StrictTrendEnv(_Strict, TrendEnv): pass
class StrictScalpEnv(_Strict, ScalpEnv): pass
if STRICT: ScalpEnv = StrictScalpEnv          # evaluate() applique aussi la règle

class PlaceboPolicy:
    """Placebo propre : à l'entrée, sens au hasard ; ensuite l'agent peut seulement garder (s'il veut ce sens) ou fermer."""
    def __init__(self, model, seed=0): self.m, self.rng = model, np.random.default_rng(seed)
    def predict(self, obs, deterministic=True):
        a = int(self.m.predict(obs, deterministic=True)[0]); pos = int(round(float(obs[-4])))
        if pos == 0: return (0 if a == 0 else int(self.rng.choice([1, 2]))), None
        return (a if (0, 1, -1)[a] == pos else 0), None

if __name__ == "__main__":
    t0 = time.time()
    F, price, atr, gap, t_dec = load()
    assert not any(c == "adv_goldstark" for c in F.columns)
    idx = lambda a, b: (int(np.searchsorted(t_dec, pd.Timestamp(a))), int(np.searchsorted(t_dec, pd.Timestamp(b))))
    tr_lo, tr_hi = idx(*TRAIN); va_lo, va_hi = idx(*VAL); te_lo, te_hi = idx(*TEST)
    Fm = mirror_features(F)
    both = np.vstack([F.values[tr_lo:tr_hi], Fm.values[tr_lo:tr_hi]])
    mu, sd = np.nanmean(both, 0), np.nanstd(both, 0) + 1e-9
    std = lambda A: np.clip(np.nan_to_num((A - mu) / sd), -10, 10).astype(np.float32)
    X, Xm = std(F.values), std(Fm.values)
    day = t_dec.normalize().values.astype("datetime64[D]").astype(np.int64)
    YEARS_ARR = np.asarray(t_dec.year); brk = gap.copy()
    h1 = np.nan_to_num(F[TREND_COLS[0]].values)
    trend = np.where(np.abs(h1) > 0.5, np.sign(h1), 0.0)                 # tendance nette : +1 / -1 / 0
    m15 = np.nan_to_num(F[TREND_COLS[1]].values)
    ALLOWED = np.where((np.abs(h1) > 0.3) & (np.sign(h1) == np.sign(m15)), np.sign(h1), 0.0)
    # poids des semaines : années récentes x (1 + TREND_SAMPLING x force de la tendance sur la semaine, en rang)
    starts = np.arange(tr_lo, max(tr_lo + 1, tr_hi - EP_LEN))
    move = np.abs(price[np.minimum(starts + EP_LEN, len(price) - 1)] - price[starts]) / np.maximum(atr[starts], 1e-6)
    w = np.exp(RECENCY * (YEARS_ARR[starts] - 2018)) * (1 + TREND_SAMPLING * pd.Series(move).rank(pct=True).values)
    cw = np.cumsum(w) / w.sum()
    Xs, Ps, Ts = [X, Xm], [price, -price], [trend, -trend]
    print(f"V5 tendance | {len(F):,} bougies x {F.shape[1]} | bonus tendance {TREND_BONUS} | tirage tendance {TREND_SAMPLING} "
          f"| bougies en tendance H1 : {np.mean(trend != 0):.0%} | STRICT {STRICT} (H1+M15 alignés : {np.mean(ALLOWED != 0):.0%} des bougies) ({(time.time()-t0)/60:.1f} min)", flush=True)

    def score(m):
        r = evaluate(m, X, price, atr, brk, day, va_lo, va_hi, t_dec)[f"cout {COST_TRAIN}$"]["total"]
        b = evaluate(m, Xm, -price, atr, brk, day, va_lo, va_hi, t_dec)[f"cout {COST_TRAIN}$"]["total"]
        return r, b
    results = []
    for seed in SEEDS:
        venv = DummyVecEnv([lambda: (StrictTrendEnv if STRICT else TrendEnv)(Xs, Ps, Ts, cw, atr, brk, day, tr_lo, tr_hi, COST_TRAIN) for _ in range(16)])
        if INIT_FROM:                                                   # on reprend l'agent existant et on continue avec la nouvelle règle
            src = next(INPUT.rglob(f"ppo_{INIT_FROM}_seed{seed}_best.zip"))
            model = PPO.load(src, env=venv, device="cpu", custom_objects={"learning_rate": 1e-4})
            print(f"  graine {seed} : reprise de {src.name}", flush=True)
        else:
          model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=4096, n_epochs=5, learning_rate=2e-4,
                      gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2,
                      policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])), verbose=0, device="cpu")
        best, chunk = None, STEPS // 10
        for k in range(10):
            venv.set_attr("cost", CURRICULUM[min(k, len(CURRICULUM) - 1)])
            model.learn(chunk, reset_num_timesteps=False)
            r, b = score(model)
            print(f"  graine {seed} | {(k+1)*chunk/1e6:.1f}M pas | VAL 2024 à {COST_TRAIN}$ : réel {r:+.0f} | miroir (baisse) {b:+.0f} "
                  f"| moyenne {(r + b) / 2:+.0f} ({(time.time()-t0)/60:.0f} min)", flush=True)
            if best is None or (r + b) / 2 > best[0]:
                best = ((r + b) / 2, k); model.save(OUT / f"ppo_{TAG}_seed{seed}_best")
        results.append((best[0], seed))
        json.dump({"seed": seed, "val_mean_015": float(best[0]), "chunk": int(best[1])}, open(OUT / f"ppo_{TAG}_seed{seed}_val.json", "w"))

    print("\n" + "=" * 100 + "\nTEST FINAL 2025-2026 (jamais vu) — " + TAG + " suivi de tendance\n" + "=" * 100)
    for _, s in sorted(results, reverse=True):
        mdl = PPO.load(OUT / f"ppo_{TAG}_seed{s}_best")
        r = evaluate(mdl, X, price, atr, brk, day, te_lo, te_hi, t_dec)
        b = evaluate(mdl, Xm, -price, atr, brk, day, te_lo, te_hi, t_dec)
        pl = [evaluate(PlaceboPolicy(mdl, q), X, price, atr, brk, day, te_lo, te_hi, t_dec)["cout 0.15$"]["total"] for q in range(3)]
        print(f"\n  graine {s} :")
        for k, v in r.items(): print(f"    {k} : {v}")
        print(f"    MIROIR (marché en baisse) : {b['trades/jour']} trades/j, BUY {b['part_BUY']} | 0,05$ -> {b['cout 0.05$']['total']} "
              f"| 0,15$ -> {b['cout 0.15$']}")
        print(f"    PLACEBO propre (sens au hasard, 3 tirages) à 0,15$ : {', '.join(f'{x:+.0f}' for x in pl)}")
    print(f"\nTerminé en {(time.time()-t0)/60:.1f} min")
