"""Prépare les notebooks Kaggle du robot RL OR M1 (mêmes assemblages que ceux utilisés en novembre 2026).

Les scripts de sources/ sont collés les uns après les autres dans un seul fichier, avec les réglages de l'entraînement
d'origine en tête (60 min max, objectif 20 trades/jour, règle stricte EMA50 M15, bonus de tendance 0,005, 12M pas).

  python preparer_kaggle.py donnees
        -> xau-scalp-phase0-costs : retélécharge les bougies Dukascopy XAUUSD M1 bid/ask de 2016 jusqu'à aujourd'hui
  python preparer_kaggle.py agent NOM GRAINE [--dates FIN_TRAIN,FIN_VAL]
        -> xau-rl-m1-NOM : entraîne UN agent (12M pas ≈ 6 h ; Kaggle coupe à 12 h et accepte 5 notebooks à la fois)
  python preparer_kaggle.py test NOM GRAINES [--dataset UTILISATEUR/NOM] [--dates ...]
        -> xau-rl-m1-NOM : relit les agents du dataset, test final, comités de 3 / 5 / 6, placebo
  python preparer_kaggle.py export [--dataset ...] [--comite 1,2,3,5,6]
        -> xau-m1-live-export : fichiers numpy du comité (norm_m1.npz, policy_m1_seed*.npz) pour Render / le PC
  python preparer_kaggle.py sorties PART
        -> xau-m1-exit-test-PART : test des sorties d'urgence (stop, tendance rapide, spread...)

--dates FIN_TRAIN,FIN_VAL : pour apprendre sur des bougies plus récentes, ex. --dates 2025-07-01,2026-04-01
  (sans l'option : apprentissage 2018-2023, choix sur 2024, test 2025-2026 = l'entraînement d'origine).
Le notebook est écrit dans entrainement/_kaggle/<nom>/ ; ensuite : kaggle kernels push -p "<ce dossier>"
"""
import argparse, json
from pathlib import Path

ICI = Path(__file__).resolve().parent
SRC, RACINE, SORTIE = ICI / "sources", ICI.parent, ICI / "_kaggle"
USER = "christianondiyo"
DONNEES = f"{USER}/xau-scalp-phase0-costs"
REGLAGES = {"GS_MAX_HOLD": "60", "GS_IDLE": "0", "GS_TREND_BONUS": "0.005", "GS_STRICT": "1",
            "GS_TREND_COLS": "m15_d_ema50,m15_d_ema50", "GS_TARGET": "20"}
DATES_ORIGINE = 'TRAIN = ("2018-01-01", "2024-01-01"); VAL = ("2024-01-01", "2025-01-01"); TEST = ("2025-01-01", "2027-01-01")'

cut = lambda t: t.split('if __name__ == "__main__":')[0]
rd = lambda p: p.read_text(encoding="utf-8")

def socle():
    """Phase 1 (indicateurs) + simulateur v4 + règles v5 + 16 stratégies M1, sans leurs parties « main »."""
    return (cut(rd(SRC / "scalp_phase1_map.py")) + "\n" + cut(rd(SRC / "scalp_rl_agent_v4.py")) + "\n"
            + cut(rd(SRC / "scalp_rl_agent_v5.py")) + "\n" + cut(rd(SRC / "scalp_m1_trend_strategies.py")) + "\n")

def ecrire(nom, fichier, env, src, datasets=(), kernels=(DONNEES,), dates=None):
    if dates:
        a, b = dates.split(",")
        assert DATES_ORIGINE in src, "ligne des dates introuvable dans scalp_rl_agent_v4.py"
        src = src.replace(DATES_ORIGINE, f'TRAIN = ("2018-01-01", "{a}"); VAL = ("{a}", "{b}"); TEST = ("{b}", "2030-01-01")')
    if env:
        src = "import os\n" + "".join(f"os.environ.setdefault({k!r}, {v!r})\n" for k, v in env.items()) + src
    compile(src, fichier, "exec")
    K = SORTIE / nom; K.mkdir(parents=True, exist_ok=True)
    (K / fichier).write_text(src, encoding="utf-8")
    (K / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{nom}", "title": nom, "code_file": fichier, "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_internet": True, "dataset_sources": list(datasets),
        "competition_sources": [], "kernel_sources": list(kernels)}, indent=2), encoding="utf-8")
    print(f"Prêt : {K}\nLancer : kaggle kernels push -p \"{K}\"")
    return K

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["donnees", "agent", "test", "export", "sorties"])
    p.add_argument("args", nargs="*")
    p.add_argument("--dates", default=None)
    p.add_argument("--dataset", default=f"{USER}/xau-rl-m1-agents")
    p.add_argument("--comite", default="1,2,3,5,6")
    a = p.parse_args()
    if a.mode == "donnees":
        ecrire("xau-scalp-phase0-costs", "scalp_phase0_costs.py", {}, rd(SRC / "scalp_phase0_costs.py"), kernels=())
    elif a.mode in ("agent", "test"):
        nom, graines = a.args[:2]
        env = {"GS_STEPS": "12000000", "GS_SEEDS": graines, **REGLAGES, "GS_EVAL_ONLY": "1" if a.mode == "test" else "0"}
        ecrire(f"xau-rl-m1-{nom}", f"xau_rl_m1_{nom}.py", env, socle() + rd(SRC / "scalp_rl_agent_m1.py"),
               datasets=[a.dataset] if a.mode == "test" else [], dates=a.dates)
    elif a.mode == "export":
        src = (socle() + cut(rd(SRC / "scalp_rl_agent_m1.py")) + "\n" + rd(RACINE / "features_v5.py") + "\n"
               + rd(RACINE / "features_m1.py") + "\n" + rd(SRC / "scalp_m1_live_export.py"))
        ecrire("xau-m1-live-export", "xau_m1_live_export.py", {**REGLAGES, "GS_COMMITTEE": a.comite}, src,
               datasets=[a.dataset], dates=a.dates)
    else:
        part = a.args[0] if a.args else ""
        env = {**REGLAGES, "GS_COMMITTEE": a.comite, **({"GS_PART": part} if part != "" else {})}
        nom = "xau-m1-exit-test" + (f"-{part}" if part != "" else "")
        ecrire(nom, "xau_m1_exit_test.py", env, socle() + cut(rd(SRC / "scalp_rl_agent_m1.py")) + "\n"
               + rd(SRC / "scalp_m1_exit_test.py"), datasets=[a.dataset], dates=a.dates)

if __name__ == "__main__":
    main()
