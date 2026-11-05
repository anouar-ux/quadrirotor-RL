"""
Export d'un agent déjà entraîné pour la mission de la Figure 1 (0 -> 0,5 -> -0,25 m), sans réentraînement.

    python export_z.py                                   # agent par défaut, écrit dans ../rl_interface/
    python export_z.py --run resultats/agent_z_ppo

policy.mat : le même réseau que pour la consigne unique (rien à réapprendre).
traj_model.mat : le vol de la MISSION sur le modèle GRU, que check_controller (EINT_RL_eval.m) compare à rl_controller.m.
Les paliers MISSION_FIG1 de env_z.py doivent être recopiés à l'identique dans rl_controller.m.
"""
import argparse
import os
import sys
from pathlib import Path

from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import PYTHON  # noqa: E402  (rend importables les sous-dossiers de python/)
HERE = str(PYTHON)           # racine python/ : resultats/ ; ../rl_interface = EI_RL/rl_interface
from env_z import MISSION_FIG1, DroneEnv  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "rl_interface"))
from export_policy import export_for_matlab  # noqa: E402  (fichier fourni, non modifié)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=os.path.join("resultats", "agent_z_ppo"), help="dossier contenant best_model.zip")
    p.add_argument("--out", default=os.path.join("..", "rl_interface"), help="dossier où écrire policy.mat et traj_model.mat")
    a = p.parse_args()

    agent = PPO.load(os.path.join(HERE, a.run, "best_model"), device="cpu")
    out = os.path.abspath(os.path.join(HERE, a.out))
    cwd = os.getcwd()
    os.chdir(out)                                                    # export_for_matlab écrit dans le dossier courant
    try:
        export_for_matlab(agent, DroneEnv(z_ref_fixe=MISSION_FIG1))   # SANS bruit, mission de la Figure 1
    finally:
        os.chdir(cwd)
    print(f"Mission {MISSION_FIG1} exportée dans {out}")


if __name__ == "__main__":
    main()
