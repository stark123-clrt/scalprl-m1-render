# Comité RL OR M1 sur Render

Même robot que `ScalpRL_Trend_M1` (comité de 5 agents 1, 2, 3, 5, 6, règle de tendance EMA50 M15),
mais le **cerveau tourne sur Render** au lieu du service Python du PC. Le PC n'a plus besoin que de MT5.

```
MT5 (EA ScalpRL_M1_Render)                         Render (app.py)
  au démarrage : historique M1  ── POST /history ─▶  garde ~115 000 minutes en mémoire
  chaque minute : état + 30 bougies ─ POST /decide ─▶  calcule les 136 informations, vote du comité,
                ◀── « id, action, votes » ──────────   règle de tendance -> BUY / SELL / à plat
  l'EA passe (ou non) l'ordre sur le compte Deriv
```

Les bougies viennent de **ton MT5** (prix, spread, volume Deriv), comme pour le robot du PC :
les décisions sont donc les mêmes (même code de calcul, copié tel quel).

## Contenu du dossier

| Fichier | Rôle |
|---|---|
| `app.py` | le service (FastAPI) : `/history`, `/decide`, `/health`, `/log` |
| `features_m1.py`, `strategies_m1.py`, `phase1.py`, `features.py`, `features_v5.py` | calcul des informations (copies exactes de `scalp_rl_live`) |
| `models_m1/` | les 5 agents + la normalisation (~4 Mo) |
| `requirements.txt`, `Dockerfile`, `render.yaml` | installation sur Render |
| `mql5/ScalpRL_M1_Render.mq5` | le nouvel EA (WebRequest), compilé sans erreur |

## Mise en ligne

1. **GitHub** : créer un dépôt **PRIVÉ** (les modèles sont dedans) et y mettre le *contenu* de ce dossier
   (`app.py` doit être à la racine du dépôt).
2. **Render** : *New > Blueprint* et choisir le dépôt (le fichier `render.yaml` règle tout),
   ou *New > Web Service > Docker*.
   - Offre **Starter** (512 Mo, 0,5 processeur, ~7 $/mois). L'offre gratuite est trop lente (0,1 processeur)
     et s'endort sans visite.
   - Variable d'environnement `SCALPRL_TOKEN` : c'est le mot de passe entre l'EA et le service
     (le Blueprint le génère tout seul ; le recopier depuis *Environment*).
   - Vérifier : `https://<nom-du-service>.onrender.com/health` doit répondre.
3. **MT5** :
   - *Outils > Options > Expert Advisors* : cocher « Autoriser WebRequest pour les URL listées »
     et ajouter `https://<nom-du-service>.onrender.com`.
   - Copier `ScalpRL_M1_Render.mq5` dans `MQL5\Experts\ScalpRL` et le compiler.
   - Ouvrir un **deuxième** graphique XAUUSD M1, y glisser `ScalpRL_M1_Render`,
     remplir `InpURL` et `InpToken`, laisser `InpShadow = true`.

## Mode fantôme (par défaut) — à faire quelques jours

L'EA demande chaque minute la décision à Render **en suivant la position du robot actuel** (magique 260501),
l'écrit dans l'onglet *Experts* (`FANTÔME | Render : BUY ...`) mais **ne passe aucun ordre**.
On compare avec le journal du robot du PC (`scalp_rl_live\logs\service_m1.log`) : les décisions doivent être identiques.
Le journal de Render est aussi visible sur `/log` (avec l'en-tête `X-Token`) et dans *Logs* sur Render.

## Passer en réel

1. Arrêter l'ancien robot : retirer `ScalpRL_Trend_M1` du graphique et fermer la fenêtre `start_service_m1.bat`.
2. Sur `ScalpRL_M1_Render` : `InpShadow = false` (numéro magique 260502, lot 0,01, SL de sécurité `InpSafetySL`).

## Bon à savoir

- **Un seul processus** (`--workers 1`) : l'historique est gardé en mémoire. Après un redéploiement ou un redémarrage
  de Render, le service répond `need_history` et l'EA renvoie l'historique tout seul (~10 s).
- Tous les soirs, à la réouverture du marché (23:05 UTC), une ligne « dernière bougie incohérente » est normale :
  pas de décision sur la toute première minute.
- Mesuré sur le PC : ~400 Mo de mémoire et 1 à 2 s par décision (fenêtre 110 000 minutes, `SCALPRL_WINDOW`).
- Le mot de passe (`SCALPRL_TOKEN`) et le dépôt doivent rester privés.
- Compte **démo** uniquement tant que le mode fantôme et quelques semaines de réel n'ont pas été vérifiés.
