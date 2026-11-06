# Entraînement du robot RL OR M1 (sauvegarde complète)

Tout ce qu'il faut pour **refaire, tester ou réentraîner** le comité M1 de l'or qui tourne sur Render.
Les fichiers de `sources/` sont les copies exactes de ceux qui ont servi en novembre 2026, et
`preparer_kaggle.py` reproduit à l'identique les notebooks Kaggle de l'époque (vérifié le 06/11/2026).

## Contenu

| Fichier | Rôle |
|---|---|
| `sources/scalp_phase0_costs.py` | télécharge les bougies Dukascopy XAUUSD M1 (bid + ask) de 2016 à aujourd'hui |
| `sources/scalp_phase1_map.py` | calcul des indicateurs (M1, M5, M15, H1, H4) |
| `sources/scalp_rl_agent_v4.py` | le simulateur de trading (coût, 60 min max, monde miroir) |
| `sources/scalp_rl_agent_v5.py` | la règle stricte de tendance |
| `sources/scalp_m1_trend_strategies.py` | les 16 stratégies M1 (signaux donnés à l'IA) |
| `sources/scalp_rl_agent_m1.py` | l'agent M1 : 136 informations, entraînement, choix sur la période VAL, test + miroir + placebo + comités |
| `sources/scalp_m1_live_export.py` | exporte le comité en fichiers numpy (`models_m1/`) et vérifie la parité |
| `sources/scalp_m1_exit_test.py` | test des sorties d'urgence (stop, tendance rapide, spread...) |
| `preparer_kaggle.py` | assemble ces fichiers en notebooks Kaggle prêts à lancer |
| `agents_complets/` | les 6 agents PPO complets (`ppo_m1_seed1..6_best.zip`, Stable-Baselines3), leurs notes et journaux d'entraînement |

Le comité en service = agents **1, 2, 3, 5, 6** (version numpy légère dans `../models_m1/`, lue par Render).

## Réglages de l'entraînement d'origine

- Données : Dukascopy XAUUSD M1 bid/ask, prix milieu, depuis 2018.
- Coût : 0,15 $ par trade (spread Deriv), appris progressivement (0 → 0,05 → 0,15).
- Apprentissage 2018-2023, choix du meilleur moment sur 2024 (marché réel + marché miroir), test 2025-2026 jamais vu.
- PPO [256, 256], 12 millions de pas par agent, 60 min max par trade, objectif doux 20 trades/jour,
  bonus de tendance 0,005, règle stricte : achat seulement si EMA50 M15 en hausse (> 0,3 ATR), vente seulement en baisse.
- Résultat test 2025-2026 à 0,15 $ du comité de 5 : **+279 $ (réel) / +252 $ (miroir)** pour 0,01 lot.

## Réentraîner avec les nouvelles bougies

Il faut `pip install kaggle` et la clé API Kaggle (`kaggle.json`) sur le PC. Kaggle : 5 notebooks à la fois, 12 h max chacun.

1. **Bougies à jour** :
   `python preparer_kaggle.py donnees` puis la commande `kaggle kernels push -p ...` affichée (≈ 1 h).
2. **Nouveaux agents** : un notebook par agent (≈ 6 h chacun). Prendre de **nouveaux noms** pour ne pas écraser les anciens,
   et décaler les dates pour que l'IA apprenne sur les bougies récentes, par exemple :
   ```
   python preparer_kaggle.py agent r1 1 --dates 2025-07-01,2026-04-01
   python preparer_kaggle.py agent r2 2 --dates 2025-07-01,2026-04-01
   ...  (graines 1 à 6)
   ```
   `--dates FIN_APPRENTISSAGE,FIN_CHOIX` : apprentissage jusqu'à la 1re date, choix du meilleur moment entre les deux,
   test après la 2e (jamais vu).
3. **Mettre les 6 `ppo_m1_seed*_best.zip`** (sorties des notebooks) dans un **nouveau dataset Kaggle privé**,
   par exemple `xau-rl-m1-agents-r1` (Kaggle décompresse les .zip : le code sait les relire).
4. **Tester et comparer, avec les mêmes dates** :
   ```
   python preparer_kaggle.py test r-test "1,2,3,4,5,6" --dataset christianondiyo/xau-rl-m1-agents-r1 --dates 2025-07-01,2026-04-01
   python preparer_kaggle.py test old-test "1,2,3,4,5,6" --dataset christianondiyo/xau-rl-m1-agents --dates 2025-07-01,2026-04-01
   ```
   Le 2e notebook teste **les agents actuels sur la même période récente** : on ne remplace le comité
   que si le nouveau fait mieux (réel ET miroir, après coût).
5. **Exporter le nouveau comité** :
   `python preparer_kaggle.py export --dataset christianondiyo/xau-rl-m1-agents-r1 --comite 1,2,3,5,6 --dates ...`,
   puis remplacer `../models_m1/` (norm + policies + `committee.json`, toujours ensemble) et pousser sur GitHub :
   Render se met à jour tout seul.

## Prudence

- Réentraîner **ne garantit pas** un meilleur robot. En octobre, reprendre les agents v5 avec une nouvelle règle les a rendus pires.
  Toujours comparer sur la même période avant de remplacer.
- `norm_m1.npz` (la mise à l'échelle des informations) change avec les données : il va toujours avec ses agents.
- Le dépôt doit rester **privé** (agents et code dedans).
