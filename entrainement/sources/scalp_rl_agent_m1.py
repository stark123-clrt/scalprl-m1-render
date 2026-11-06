
# ════════════ AGENT RL OR M1 : SUIVI DE TENDANCE (comité M1, demande de l'utilisateur 03/11/2026) ════════════
# Chargé après : phase 1 (ohlc, ind, TF), agent v4 (ScalpEnv, evaluate, PPO...), agent v5 (TrendEnv, _Strict, règle
# stricte) et scalp_m1_trend_strategies.py (16 stratégies YouTube/TradingView : fonction signals).
# Comme le robot or M5 v5 strict, MAIS décision à la CLÔTURE DE CHAQUE BOUGIE M1 :
#   - observation : indicateurs M1 + M5 / M15 / H1 / H4 FERMÉES, 10 dernières bougies M1, spread, heure,
#     les 16 stratégies M1 (+ leur bilan des 5 dernières minutes), état de la position ;
#   - règle stricte de tendance = celle du robot M5 actuel (R4 : prix vs EMA50 M15, > 0,3 ATR) ;
#   - 60 min max par position, coût progressif 0 -> 0,15 $, marché miroir, semaines en tendance favorisées ;
#   - choix du meilleur entraînement sur 2024 réel + miroir ; TEST final 2025-2026 + miroir + placebo.
# Évaluation rapide : le réseau de l'agent est recopié en numpy (des centaines de milliers de bougies M1 à rejouer).
EP_M1 = int(os.environ.get("GS_EP_M1", 2880))                 # ~2 jours de bougies M1 par épisode
START_M1 = os.environ.get("GS_START_M1", "2018-01-01")

def load_m1_agent():
    m = pd.read_parquet(next(INPUT.rglob("xau_m1_bidask.parquet"))).set_index("datetime").sort_index()
    m = m[~m.index.duplicated()]
    m1 = pd.DataFrame({"o": (m.o_bid + m.o_ask) / 2, "h": (m.h_bid + m.h_ask) / 2, "l": (m.l_bid + m.l_ask) / 2,
                       "c": (m.c_bid + m.c_ask) / 2, "spread": m.spread, "v": m.v})
    del m
    STR = signals(m1[["o", "h", "l", "c", "v"]])                       # sur tout l'historique (indicateurs chauds)
    m1 = m1[m1.index >= pd.Timestamp(START_M1) - pd.Timedelta(days=30)]
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
    print("  stratégies M1 (signaux/jour moyens) : " + ", ".join(f"{c[:3]} {np.mean(STR[c] != 0) * 1380:.0f}" for c in STR.columns), flush=True)
    F = pd.concat([F, STR, rec], axis=1).replace([np.inf, -np.inf], np.nan)
    keep = F.index >= pd.Timestamp(START_M1)
    F, price, atr_a, t_dec = F[keep], m1.c.values[keep], atr1.values[keep], t_dec[keep]
    gap = np.r_[np.diff(F.index.values) > np.timedelta64(15, "m"), True]           # nuit / week-end après la bougie
    return F, price, atr_a, gap, t_dec

def mirror_features(F):
    M = F.copy()
    for c in F.columns:
        if any(k in c for k in ("_ret1", "_ret3", "_ret_l", "_body", "_d_ema", "_macd_h", "_bb", "_di")) \
                or (len(c) > 3 and c[0] == "y" and c[1:3].isdigit() and c[3] == "_"):      # stratégies : BUY <-> SELL
            M[c] = -F[c]
        elif c.endswith("_rsi"):
            M[c] = 100 - F[c]
    for c in F.columns:
        if c.endswith("_wick_up"): M[c] = F[c.replace("_wick_up", "_wick_dn")]
        if c.endswith("_wick_dn"): M[c] = F[c.replace("_wick_dn", "_wick_up")]
    return M

class FastPolicy:
    """Copie numpy du réseau de décision PPO (MlpPolicy, tanh) : même action déterministe, ~50x plus rapide."""
    def __init__(self, model):
        net = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
        self.W = [(l.weight.detach().numpy().T.copy(), l.bias.detach().numpy().copy()) for l in net]
        a = model.policy.action_net; self.A = (a.weight.detach().numpy().T.copy(), a.bias.detach().numpy().copy())
    def predict(self, obs, deterministic=True):
        x = obs
        for w, b in self.W: x = np.tanh(x @ w + b)
        return int(np.argmax(x @ self.A[0] + self.A[1])), None

class FastCommittee:
    def __init__(self, models): self.ms = [FastPolicy(m) for m in models]
    def predict(self, obs, deterministic=True):
        votes = np.bincount([m.predict(obs)[0] for m in self.ms], minlength=3); best = np.flatnonzero(votes == votes.max())
        return (int(best[0]) if len(best) == 1 else 0), None

if __name__ == "__main__":
    t0 = time.time()
    F, price, atr_a, gap, t_dec = load_m1_agent()
    idx = lambda a, b: (int(np.searchsorted(t_dec, pd.Timestamp(a))), int(np.searchsorted(t_dec, pd.Timestamp(b))))
    tr_lo, tr_hi = idx(*TRAIN); va_lo, va_hi = idx(*VAL); te_lo, te_hi = idx(*TEST)
    Fm = mirror_features(F)
    sub = np.arange(tr_lo, tr_hi, 7)                                    # normalisation sur un échantillon (mémoire)
    both = np.vstack([F.values[sub], Fm.values[sub]])
    mu, sd = np.nanmean(both, 0), np.nanstd(both, 0) + 1e-9; del both
    std = lambda A: np.clip(np.nan_to_num((A - mu) / sd), -10, 10).astype(np.float32)
    X, Xm = std(F.values), std(Fm.values)
    trend_src = np.nan_to_num(F[TREND_COLS[0]].values); trend_src2 = np.nan_to_num(F[TREND_COLS[1]].values)
    cols = list(F.columns); del Fm
    day = t_dec.normalize().values.astype("datetime64[D]").astype(np.int64)
    YEARS_ARR = np.asarray(t_dec.year); brk = gap.copy()
    trend = np.where(np.abs(trend_src) > 0.5, np.sign(trend_src), 0.0)
    ALLOWED = np.where((np.abs(trend_src) > 0.3) & (np.sign(trend_src) == np.sign(trend_src2)), np.sign(trend_src), 0.0)
    starts = np.arange(tr_lo, max(tr_lo + 1, tr_hi - EP_M1))
    move = np.abs(price[np.minimum(starts + EP_M1, len(price) - 1)] - price[starts]) / np.maximum(atr_a[starts], 1e-6)
    w = np.exp(RECENCY * (YEARS_ARR[starts] - 2018)) * (1 + TREND_SAMPLING * pd.Series(move).rank(pct=True).values)
    cw = np.cumsum(w) / w.sum()
    Xs, Ps, Ts = [X, Xm], [price, -price], [trend, -trend]
    print(f"M1 tendance | {len(F):,} bougies x {len(cols)} | règle {TREND_COLS} (tendance active {np.mean(ALLOWED != 0):.0%}) | "
          f"max {MAX_HOLD} min | objectif {TARGET} trades/j | {STEPS/1e6:.0f}M pas ({(time.time()-t0)/60:.1f} min)", flush=True)

    def ev(m, lo, hi, mirror=False):
        return evaluate(FastPolicy(m) if not isinstance(m, (FastPolicy, FastCommittee)) else m,
                        Xm if mirror else X, -price if mirror else price, atr_a, brk, day, lo, hi, t_dec)
    def placebo(m, lo, hi, c=0.15, n=1000):
        """Placebo propre : on rejoue l'agent puis on garde SES trades (mêmes entrées, mêmes sorties) en tirant seulement
        le sens au hasard, n fois. Renvoie la moyenne des tirages et la part des tirages qui font aussi bien que l'agent.
        (Un placebo qui pilote l'agent en inversant son sens transformait ses demandes interdites par la règle en trades.)"""
        env = ScalpEnv([X], [price], atr_a, brk, day, lo, hi, 0.0, random_start=False)
        obs, _ = env.reset(); done = False
        while not done: obs, _, done, _, _ = env.step(m.predict(obs, deterministic=True)[0])
        g = np.array(env.trades); tot = (np.random.default_rng(0).choice([-1, 1], (n, len(g))) * g).sum(1) - c * len(g)
        return tot.mean(), np.mean(tot >= g.sum() - c * len(g))
    results = []
    EVAL_ONLY = os.environ.get("GS_EVAL_ONLY") == "1"
    if EVAL_ONLY:    # agents déjà entraînés (notebooks coupés à 12 h) : relus, classés sur 2024
        import re, shutil
        found = sorted(INPUT.rglob("ppo_m1_seed*_best.zip")) + sorted(q.parent for q in INPUT.rglob("policy.pth"))
        for p in found:                          # un dataset Kaggle décompresse les .zip : dossier -> on refait le zip
            m_ = re.search(r"seed(\d+)_best", str(p))
            if not m_ or int(m_.group(1)) in [x for _, x in results]: continue
            s, z = int(m_.group(1)), OUT / f"ppo_m1_seed{int(m_.group(1))}_best.zip"
            if p.is_dir(): shutil.make_archive(str(z)[:-4], "zip", p)
            else: shutil.copy(p, z)
            mdl = PPO.load(z, device="cpu")
            r, b = ev(mdl, va_lo, va_hi), ev(mdl, va_lo, va_hi, True)
            rt, bt = r[f"cout {COST_TRAIN}$"]["total"], b[f"cout {COST_TRAIN}$"]["total"]
            print(f"  graine {s} relue | VAL 2024 à {COST_TRAIN}$ : réel {rt:+.0f} ({r['trades/jour']} tr/j) | miroir {bt:+.0f}", flush=True)
            results.append(((rt + bt) / 2, s))
        assert results, f"aucun agent trouvé dans {INPUT} : {[str(q) for q in list(INPUT.rglob('*'))[:40]]}"
    for seed in ([] if EVAL_ONLY else SEEDS):
        venv = DummyVecEnv([lambda: StrictTrendEnv(Xs, Ps, Ts, cw, atr_a, brk, day, tr_lo, tr_hi, COST_TRAIN, ep_len=EP_M1) for _ in range(16)])
        model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=4096, n_epochs=5, learning_rate=2e-4,
                    gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])), verbose=0, device="cpu")
        best, chunk = None, STEPS // 10
        for k in range(10):
            venv.set_attr("cost", CURRICULUM[min(k, len(CURRICULUM) - 1)])
            model.learn(chunk, reset_num_timesteps=False)
            r, b = ev(model, va_lo, va_hi), ev(model, va_lo, va_hi, True)
            rt, bt = r[f"cout {COST_TRAIN}$"]["total"], b[f"cout {COST_TRAIN}$"]["total"]
            print(f"  graine {seed} | {(k+1)*chunk/1e6:.1f}M pas | VAL 2024 à {COST_TRAIN}$ : réel {rt:+.0f} ({r['trades/jour']} tr/j) "
                  f"| miroir {bt:+.0f} | moyenne {(rt + bt) / 2:+.0f} ({(time.time()-t0)/60:.0f} min)", flush=True)
            if best is None or (rt + bt) / 2 > best[0]:
                best = ((rt + bt) / 2, k); model.save(OUT / f"ppo_m1_seed{seed}_best")
        results.append((best[0], seed))
        json.dump({"seed": seed, "val_mean_015": float(best[0]), "chunk": int(best[1])}, open(OUT / f"ppo_m1_seed{seed}_val.json", "w"))

    print("\n" + "=" * 100 + "\nTEST FINAL 2025-2026 (jamais vu) — agents M1 suivi de tendance (durée en minutes = duree_moy_min / 5)\n" + "=" * 100)
    models = []
    for _, s in sorted(results, reverse=True):
        mdl = PPO.load(OUT / f"ppo_m1_seed{s}_best", device="cpu"); models.append(mdl)
        r, b = ev(mdl, te_lo, te_hi), ev(mdl, te_lo, te_hi, True)
        pm, pp = placebo(FastPolicy(mdl), te_lo, te_hi)
        print(f"\n  graine {s} :")
        for k, v in r.items(): print(f"    {k} : {v}")
        print(f"    MIROIR (baisse) : {b['trades/jour']} tr/j, BUY {b['part_BUY']} | 0,15$ -> {b['cout 0.15$']}")
        print(f"    PLACEBO (mêmes trades, sens au hasard, 1000 tirages) à 0,15$ : moyenne {pm:+.0f} | tirages aussi bons que l'agent : {pp:.0%}")
    for n in sorted({min(3, len(models)), min(5, len(models)), len(models)}):   # les n meilleures sur 2024
        if n < 2: continue
        com = FastCommittee(models[:n]); r, b = ev(com, te_lo, te_hi), ev(com, te_lo, te_hi, True)
        pm, pp = placebo(com, te_lo, te_hi)
        print(f"\n  COMITÉ des {n} meilleures graines : réel {r['trades/jour']} tr/j -> {r['cout 0.15$']} | miroir -> {b['cout 0.15$']}"
              f" | placebo moyenne {pm:+.0f}, tirages aussi bons : {pp:.0%}")
    print(f"\nTerminé en {(time.time()-t0)/60:.1f} min")
