# Guide d'utilisation : quel fichier lancer ?

Toutes les commandes Python se lancent **depuis la racine du dépôt**, ou depuis n'importe quel autre dossier : les
chemins sont gérés par `python/chemins.py`. Les scripts MATLAB se lancent par F5 dans l'éditeur, ou par
`run('chemin/du/script.m')` : chacun ajoute lui-même les chemins dont il a besoin (`Matlab files/`, son dossier).

Les agents, les modèles identifiés et les exports sont **déjà dans le dépôt** : on peut directement vérifier, voler et
comparer, sans rien réentraîner.

## 0. Installer et vérifier

| je veux… | lancer | durée |
|---|---|---|
| installer l'environnement Python | `python -m venv .venv` puis `.venv\Scripts\pip install -r requirements.txt` | 5 min |
| vérifier tout le MATLAB (modèles LQI, 5 étapes RL : MATLAB = Python, modèles reconstruits, courts vols) | MATLAB : `verifier_installation.m` | ≈ 3 min |
| vérifier les environnements Python | `python python/tests/test_env_z_3a.py` (et les autres `test_*.py` du dossier) | quelques s chacun |

## 1. Commande par modèle (LQI « double intégrateur », LQ sur ARX) — MATLAB

| je veux… | lancer | remarque |
|---|---|---|
| calculer les gains LQI (z, ψ, θ/φ, PD de position) | `commande_LQ/gains_LQI.m` | **à lancer avant** d'ouvrir un `test_LQI_*.slx` |
| tester le LQI d'un axe | `gains_LQI.m`, puis ouvrir et lancer `commande_LQ/test_LQI_z.slx` (ou `_psi`, `_x`, `_y`) | solveur à pas variable (continu) |
| faire la mission LQ (voiture rectiligne) | `commande_LQ/mission_LQ/essai_mission_LQ.m` | 3 niveaux de bruit → `mission_LQ.png` |
| synthétiser le LQ d'attitude sur l'ARX identifié | `commande_LQ/LQ_ARX_attitude/gains_LQ_ARX_attitude.m` | → `gains_LQ_ARX_attitude.mat` |
| faire voler ce LQ d'attitude sur UAV_model | `commande_LQ/LQ_ARX_attitude/test_LQ_ARX_attitude.m` | échelons de θ et φ |
| identifier les gains k (double intégrateur) | `python python/identification/identification_gain_k.py` | z, ψ, θ |
| synthétiser les LQ sur ARX de z et du lacet (Python) | `python python/identification/lq_arx.py` | → `python/modeles/z/arx_z.json` |
| enregistrer les réponses Simulink de mes LQI (référence) | MATLAB : `commande_LQ/reponses_LQI_simulink.m` | sans bruit → `reponses_LQI_simulink.mat` |
| réajuster mes gains LQI pour un contrôleur à 20 Hz | `python python/identification/reajuster_gains_LQI.py` (après le script précédent) | → `python/modeles/gains_LQI_20Hz.json` |

## 2. Données et identification

| je veux… | lancer | durée |
|---|---|---|
| générer les données du lacet / de l'attitude | MATLAB : `generation_donnees/generation_donnees_psi.m` / `generation_donnees_attitude.m` | 5 à 20 min |
| refaire les essais d'interface (signes des moteurs) | MATLAB : `generation_donnees/essais_interface/essais_interface.m` | 1 min |
| explorer les données de z (retard, bruit) | `python python/identification/analyse_donnees_z.py` | 1 min |
| identifier le GRU de z | `python python/identification/identification_z.py` | ≈ 1 min avec GPU (plus sur CPU) |
| identifier le GRU et l'ARX du lacet | `python python/identification/identification_psi.py` | quelques min |
| identifier l'ARX de l'attitude et la translation | `python python/identification/identification_attitude.py` | 1 min |

## 3. Entraîner les agents (facultatif : les agents retenus sont fournis)

| agent | lancer | durée | sortie |
|---|---|---|---|
| z, PPO (retenu) | `python python/entrainement/train_ppo_z.py` | 1,5 min | `python/resultats/agent_z_ppo_<date>/` |
| z, SAC (historique) | `python python/entrainement/train_sac_z.py` | 28 min | `agent_z_sac_<date>/` |
| lacet, PPO | `python python/entrainement/train_ppo_psi.py` | 3,6 min | `agent_psi_ppo_<date>/` |
| x, PPO (premier) | `python python/entrainement/train_ppo_x.py` | 7 min | `agent_x_ppo_<date>/` |
| x, PPO réentraîné aux saturations (retenu) | `python python/entrainement/train_ppo_x_saturation.py` | 13 min | `agent_x_ppo_saturation_<date>/` |

Pour **adopter** un nouvel agent, renommer son dossier comme l'agent retenu (par exemple `agent_z_ppo`), ou modifier
son chemin dans `python/chemins.py`, puis refaire l'export correspondant (§ 5).

## 4. Évaluer en Python (modèles identifiés)

| je veux… | lancer | durée | sortie (`python/resultats/…`) |
|---|---|---|---|
| **comparer LQI / LQ ARX / RL** sur tous les axes et sur la mission | `python python/evaluation/comparaisons_LQI_ARX_RL.py` | ≈ 6 min | `comparaisons/` (6 figures, tableau `.md`) |
| les réponses à des PRBS longues | `python python/evaluation/essais_prbs.py` (`--court` : essais courts) | ≈ 10 min | `essais_prbs_long/` |
| la référence PD de x et ses scénarios | `python python/evaluation/reference_pd_x.py` | 1 min | `reference_pd_x/` |
| la position (x, y) avec rotation par ψ | `python python/environnements/sim_xy.py` puis `python python/evaluation/essais_prbs_xy.py` | 2 min | `essais_xy/` |
| la mission complète (nominale + 12 voitures aléatoires) | `python python/environnements/sim_mission.py` | 2 min | `mission/` |
| l'effet de la consigne d'approche (τ) et du cap | `python python/evaluation/comparaison_approche_mission.py` | 5 min | `mission/` |
| l'animation GIF de la poursuite (vol Simulink enregistré) | `python python/evaluation/animation_mission.py` | 1 min | `mission/animation_mission.gif` |

## 4 bis. Animations et figures de la présentation (sorties : `rapport/presentation/`)

| je veux… | lancer | durée |
|---|---|---|
| les GIF de suivi d'une voiture à forts à-coups (ligne droite, slalom) et les duels RL / LQ ARX / LQI | `python python/presentation/suivi_voiture.py` (`--rapide` : essai) | ≈ 8 min |
| la cible d'atterrissage (13 voitures × 3 familles) | `python python/presentation/cible_atterrissage.py` (`--figure` : redessiner seulement) | ≈ 3 min |
| le GIF « sous le capot » (inclinaison, moteurs) | `python python/presentation/sous_le_capot.py` | 1 min |
| le schéma d'architecture | `python python/presentation/architecture.py` | quelques s |
| le GIF « l'agent apprend » (réentraîne z avec étapes enregistrées, sans toucher l'agent retenu) | `python python/presentation/agent_apprend.py` (`--reentrainer`) | 5 à 15 min |

## 5. Exporter vers MATLAB puis voler sur Simulink

Chaque export écrit les politiques (`policy_*.mat`) et un vol Python de référence (`traj_*.mat`). Le script MATLAB
**vérifie** d'abord que son contrôleur rejoue ce vol à l'identique, **puis** il vole.

| étape | 1. export Python | 2. vol Simulink (MATLAB) | figures |
|---|---|---|---|
| z (mission de la Figure 1) | `python python/export/export_z.py` | `simulink_RL/z/vols_z.m` (ou `rl_interface/EINT_RL_eval.m`, fourni) | `python simulink_RL/z/graphes_z.py` |
| z + lacet (scan d'un tour) | `python python/export/export_zpsi.py` | `simulink_RL/zpsi/EINT_RL_eval_zpsi.m` (1 vol) ou `vols_zpsi.m` (6 vols) | `python simulink_RL/zpsi/graphes_zpsi.py` |
| z + lacet + x | `python python/export/export_x.py` | `simulink_RL/x/EINT_RL_eval_x.m` | `vol_x_simulink.png` |
| position (x, y) | `python python/export/export_xy.py` | `simulink_RL/xy/EINT_RL_eval_xy.m` | `vol_xy_simulink.png` |
| **mission finale** | `python python/export/export_mission.py` | `simulink_RL/mission/EINT_RL_eval_mission.m` (agents et PD, avec et sans bruit) | `graphes_mission.m` → `mission_simulink_cap.png` |

Le modèle `EINT_RL_closed_loop_<étape>.slx` est créé par `construire_boucle_<étape>.m` s'il manque.

## 6. Réglages courants

| je veux changer… | où |
|---|---|
| le bruit des vols Simulink | `g_noise` (ou la boucle `for g = [0 0.1]`) dans le script `EINT_RL_eval_*.m` ou `vols_*.m` |
| la mission finale (altitude, durée du scan, tolérances, descente) | constantes en tête de `python/environnements/sim_mission.py` **et** de `simulink_RL/mission/rl_controller_mission.m` (à garder identiques), puis export et vol |
| la trajectoire de la voiture | `VOITURE_MISSION` dans `python/environnements/voiture.py`, puis `export_mission.py` |
| le cap pendant la poursuite ou la correction du biais de l'agent x | `CONFIG_FINALE` (Python) et `CAP_VOITURE`, `CORR_BIAIS` (`rl_controller_mission.m`) |
| la mission LQ (voiture, altitude, scan) | `commande_LQ/mission_LQ/parametres_mission_LQ.m` et le bloc `mission` du modèle |
