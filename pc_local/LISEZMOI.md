# Version PC d'origine du robot OR M1 (référence)

C'est la première version du robot, avant Render : le cerveau tournait sur le PC (`service_m1.py`)
et l'EA `ScalpRL_Trend_M1.mq5` (magique 260501) échangeait avec lui par fichiers
(`MQL5\Files` commun, dossier `scalp_rl_m1`). Elle a été remplacée le 06/11/2026 par Render + `ScalpRL_M1_Render`.

Pour la relancer un jour : mettre ces fichiers avec `features.py`, `features_v5.py`, `features_m1.py`,
`strategies_m1.py` et `models_m1/` de la racine du dépôt (`phase1.py` et `service.py` d'ici, les versions complètes,
qui demandent aussi `lightgbm`), puis lancer `start_service_m1.bat`. L'original se trouve dans `goldstark\scalp_rl_live`.
