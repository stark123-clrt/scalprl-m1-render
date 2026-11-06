
# ════════════ EXPORT DU COMITÉ OR M1 POUR LE DIRECT + VÉRIFICATIONS ════════════
# Chargé après phase 1, agent v4, agent v5, stratégies M1, agent M1, scalp_rl_live/features_v5.py et features_m1.py.
# Produit : norm_m1.npz, policy_m1_seed{N}.npz (agents du comité), committee.json.
# Vérifie : 1) agent_features_m1 sur une fenêtre limitée (comme en direct) ≈ load_m1_agent (historique complet), pour
#              plusieurs tailles de fenêtre et 5 moments de 2025-2026, ET accord des décisions du comité ;
#           2) règle de tendance du direct == entraînement ; 3) numpy == PPO.predict ; temps de calcul ; spread médian.
import re, shutil
COMMITTEE = [int(s) for s in os.environ.get("GS_COMMITTEE", "1,2,3,5,6").split(",")]
WINDOWS = [int(w) for w in os.environ.get("GS_WINDOWS", "60000,100000,150000,250000").split(",")]

def agent_zip(s):
    """Agent PPO de la graine s (un dataset Kaggle décompresse les .zip en dossiers -> on refait le zip)."""
    for p in sorted(INPUT.rglob(f"ppo_m1_seed{s}_best*")):
        z = OUT / f"ppo_m1_seed{s}_best.zip"
        if p.is_dir(): shutil.make_archive(str(z)[:-4], "zip", p); return z
        if p.suffix == ".zip": shutil.copy(p, z); return z
    raise FileNotFoundError(f"agent {s} introuvable dans {INPUT}")

def mlp_votes(Ws, obs):
    """Actions des agents exportés (numpy, MLP tanh 256x256), une colonne par agent."""
    return np.stack([np.argmax(np.tanh(np.tanh(obs @ W["w0"].T + W["b0"]) @ W["w1"].T + W["b1"]) @ W["wa"].T + W["ba"], 1)
                     for W in Ws], 1)

def committee_votes(Ws, obs):
    """Vote majoritaire ; égalité -> à plat (comme features.committee_action du service)."""
    v = mlp_votes(Ws, obs); cnt = np.stack([(v == k).sum(1) for k in range(3)], 1)
    return np.where((cnt == cnt.max(1, keepdims=True)).sum(1) > 1, 0, cnt.argmax(1))

if __name__ == "__main__":
    t0 = time.time()
    F, price, atr_a, gap, t_dec = load_m1_agent()
    cols = list(F.columns)
    idx = lambda a, b: (int(np.searchsorted(t_dec, pd.Timestamp(a))), int(np.searchsorted(t_dec, pd.Timestamp(b))))
    tr_lo, tr_hi = idx(*TRAIN); te_lo, te_hi = idx(*TEST)
    Fm = mirror_features(F)
    sub = np.arange(tr_lo, tr_hi, 7)                                    # même normalisation qu'à l'entraînement
    both = np.vstack([F.values[sub], Fm.values[sub]])
    mu, sd = np.nanmean(both, 0), np.nanstd(both, 0) + 1e-9; del both, Fm
    std = lambda A: np.clip(np.nan_to_num((A - mu) / sd), -10, 10).astype(np.float32)
    X = std(F.values)
    np.savez(OUT / "norm_m1.npz", mu=mu, sd=sd, cols=np.array(cols))
    print(f"{len(cols)} features ({(time.time()-t0)/60:.1f} min)", flush=True)

    Ws, rng = [], np.random.default_rng(0)
    for s in COMMITTEE:
        mdl = PPO.load(agent_zip(s), device="cpu"); pol = mdl.policy
        pn = [l for l in pol.mlp_extractor.policy_net if isinstance(l, torch.nn.Linear)]
        W = {k: v.detach().numpy().astype(np.float64) for k, v in dict(w0=pn[0].weight, b0=pn[0].bias, w1=pn[1].weight, b1=pn[1].bias,
                                                                       wa=pol.action_net.weight, ba=pol.action_net.bias).items()}
        np.savez(OUT / f"policy_m1_seed{s}.npz", **W); Ws.append(W)
        rows = rng.integers(te_lo, te_hi, 3000)
        st = np.c_[rng.integers(-1, 2, 3000), rng.normal(0, 1, 3000), rng.integers(0, MAX_HOLD, 3000) / MAX_HOLD, rng.integers(0, 60, 3000) / TARGET]
        obs = np.c_[X[rows], st].astype(np.float32)
        a_n = mlp_votes([W], obs.astype(np.float64))[:, 0]
        print(f"[3] agent {s} : accord numpy/PPO {np.mean(mdl.predict(obs, deterministic=True)[0] == a_n):.4%}", flush=True)

    ts = np.nan_to_num(F[TREND_COLS[0]].values); ts2 = np.nan_to_num(F[TREND_COLS[1]].values)
    al_train = np.where((np.abs(ts) > 0.3) & (np.sign(ts) == np.sign(ts2)), np.sign(ts), 0.0)
    sample = rng.integers(te_lo, te_hi, 20000)
    al_live = np.array([allowed_direction(ts[i], ts2[i]) for i in sample])
    print(f"[2] règle de tendance direct == entraînement : {np.mean(al_live == al_train[sample]):.4%}", flush=True)

    m = pd.read_parquet(next(INPUT.rglob("xau_m1_bidask.parquet"))).set_index("datetime").sort_index()
    m = m[~m.index.duplicated()]
    m1 = pd.DataFrame({"o": (m.o_bid + m.o_ask) / 2, "h": (m.h_bid + m.h_ask) / 2, "l": (m.l_bid + m.l_ask) / 2,
                       "c": (m.c_bid + m.c_ask) / 2, "spread": m.spread, "v": m.v}); del m
    print(f"    spread médian Dukascopy : 20 000 dernières minutes {m1.spread.iloc[-20_000:].median():.3f} $ | "
          f"2025-2026 {m1.spread[m1.index >= pd.Timestamp(TEST[0])].median():.3f} $", flush=True)
    ends = F.index[np.linspace(te_lo + 5000, te_hi - 1, 5).astype(int)]
    for Wn in WINDOWS:
        dmax, dmean, agree, agree0, worst, secs = [], [], [], [], {}, []
        for e in ends:
            win = m1[m1.index <= e].iloc[-Wn:]                         # direct : m1[index < t_dec].iloc[-WINDOW:]
            t1 = time.time(); Fw, _, _ = agent_features_m1(win); secs.append(time.time() - t1)
            Fw = Fw[cols].iloc[-1000:]
            xa, xb = std(Fw.values), std(F.loc[Fw.index].values)
            dz = np.abs(xa - xb); dmax.append(dz.max()); dmean.append(dz.mean())
            for j in np.argsort(dz.max(0))[::-1][:3]: worst[cols[j]] = max(worst.get(cols[j], 0), float(dz[:, j].max()))
            st = np.c_[rng.integers(-1, 2, len(xa)), rng.normal(0, 1, len(xa)), rng.integers(0, MAX_HOLD, len(xa)) / MAX_HOLD,
                       rng.integers(0, 60, len(xa)) / TARGET]
            f64 = lambda A: A.astype(np.float32).astype(np.float64)
            agree.append(np.mean(committee_votes(Ws, f64(np.c_[xa, st])) == committee_votes(Ws, f64(np.c_[xb, st]))))
            z0 = np.zeros((len(xa), 4))
            agree0.append(np.mean(committee_votes(Ws, f64(np.c_[xa, z0])) == committee_votes(Ws, f64(np.c_[xb, z0]))))
        top = sorted(worst.items(), key=lambda kv: -kv[1])[:5]
        print(f"[1] fenêtre {Wn:,} min : écart max {max(dmax):.3f}, moyen {np.mean(dmean):.5f} | décisions identiques "
              f"{np.mean(agree):.2%} (à plat : {np.mean(agree0):.2%}) | calcul {np.mean(secs):.1f} s | pires : "
              + ", ".join(f"{c} {v:.3f}" for c, v in top), flush=True)

    json.dump({"seeds": COMMITTEE, "trend_rule": TREND_COLS, "max_hold": MAX_HOLD, "target": TARGET,
               "note": "comité de 5 agents OR M1 (1, 2, 3, 5, 6) choisi le 04/11/2026 ; test 2025-2026 à 0,15 $ : "
                       "+279 $ hausse / +252 $ baisse (2024 perdant : à tester en démo seulement)"},
              open(OUT / "committee.json", "w"), indent=2)
    print(f"\nTerminé en {(time.time()-t0)/60:.1f} min")
