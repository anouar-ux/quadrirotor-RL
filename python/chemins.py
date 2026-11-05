"""
CHEMINS DU PROJET, en un seul endroit, et accès aux modules entre sous-dossiers.

À quoi sert ce fichier : chaque script Python du dossier python/ commence par

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/
    from chemins import PYTHON, RESULTATS, ...

L'import de chemins.py ajoute au chemin de recherche les sous-dossiers identification/, environnements/,
entrainement/, evaluation/ et export/ : les modules s'importent alors par leur nom (from env_z import DroneEnv),
quel que soit le dossier d'où l'on lance le script.

Organisation (voir README.md à la racine) :
    EI_RL/
    ├── Matlab files/      fichiers FOURNIS (UAV_model.slx, random_init.m...)
    ├── rl_interface/      fichiers FOURNIS de la piste RL (export_policy.py, EINT_RL_eval.m...)
    ├── donnees/           données d'identification (z : fournies ; lacet et attitude : générées)
    ├── commande_LQ/       commande par modèle (MATLAB) : LQI double intégrateur, LQ sur modèle ARX, mission LQ
    ├── simulink_RL/       vols Simulink des agents RL (z, zpsi, x, xy, mission)
    └── python/
        ├── identification/  modèles des dynamiques (GRU de z et du lacet, ARX de l'attitude, gains k)
        ├── environnements/  environnements Gymnasium et simulateurs (z, psi, x, xy, mission, voiture)
        ├── entrainement/    entraînements PPO / SAC des agents
        ├── evaluation/      essais PRBS, référence PD, comparaisons LQI / ARX / RL, animation
        ├── export/          export des agents vers simulink_RL/ et rl_interface/
        ├── presentation/    animations et figures de la présentation finale (-> rapport/presentation/)
        ├── tests/           tests des environnements
        ├── modeles/         modèles identifiés (z, psi, attitude)
        └── resultats/       agents entraînés et résultats des essais
"""
import sys
from pathlib import Path

PYTHON = Path(__file__).resolve().parent          # EI_RL/python/
EI = PYTHON.parent                                # EI_RL/
MODELES = PYTHON / "modeles"
RESULTATS = PYTHON / "resultats"
DONNEES = EI / "donnees"
MATLAB = EI / "Matlab files"                      # fichiers fournis (UAV_model, random_init)
RL_INTERFACE = EI / "rl_interface"                # fichiers fournis de la piste RL
SIMULINK_RL = EI / "simulink_RL"                  # vols Simulink des agents (z, zpsi, x, xy, mission)
COMMANDE_LQ = EI / "commande_LQ"                  # commande par modèle (MATLAB)
PRESENTATION = EI / "rapport" / "presentation"    # animations et figures de la présentation finale

# Agents RETENUS (un dossier par agent : best_model.zip, results.json, figures)
AGENT_Z = RESULTATS / "agent_z_ppo"               # altitude, PPO
AGENT_Z_SAC = RESULTATS / "agent_z_sac"           # altitude, SAC (premier essai, historique)
AGENT_PSI = RESULTATS / "agent_psi_ppo"           # lacet, PPO
AGENT_X_V1 = RESULTATS / "agent_x_ppo"            # position x, PPO, premier entraînement
AGENT_X = RESULTATS / "agent_x_ppo_saturation"    # position x (et y), PPO réentraîné aux saturations : RETENU

for _d in ("identification", "environnements", "entrainement", "evaluation", "export", "presentation"):
    _p = str(PYTHON / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
