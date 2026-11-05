"""
Export des deux agents (z et lacet) pour la tâche avancée, vers simulink_RL/zpsi/ (contrôleur controleur_zpsi.m).

    python export_zpsi.py                               # agent z retenu + dernier agent de lacet
    python export_zpsi.py --lacet resultats/agent_psi_ppo_AAAAMMJJ_HHMM

policy_z.mat, policy_psi.mat : les acteurs, au même format que policy.mat (fonction _policy_matrices du fichier
fourni export_policy.py, utilisée sans le modifier).
traj_z.mat, traj_psi.mat : 400 pas (20 s) de vol SANS bruit sur les modèles GRU, avec les missions MISSION_AVANCEE_Z et
MISSION_SCAN ; EINT_RL_eval_zpsi.m les rejoue dans controleur_zpsi.m pour vérifier qu'il calcule la même chose.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import PYTHON  # noqa: E402  (rend importables les sous-dossiers de python/)
HERE = PYTHON                # racine python/ : resultats/ ; HERE.parent = EI_RL/
from env_z import MISSION_AVANCEE_Z, DroneEnv  # noqa: E402
from env_psi import MISSION_SCAN, N_STEPS, YawEnv  # noqa: E402
sys.path.insert(0, str(HERE.parent / "rl_interface"))
from export_policy import _policy_matrices  # noqa: E402  (fichier fourni, non modifié)


def vol(agent, env, cles):
    obs, info = env.reset(seed=0)
    log = {k: [] for k in (*cles, "obs", "a")}
    for _ in range(N_STEPS):
        a = float(np.asarray(agent.predict(obs, deterministic=True)[0]).reshape(-1)[0])
        for k in cles:
            log[k].append(float(info["m"][k]))
        log["obs"].append(np.asarray(obs, np.float64)); log["a"].append(a)
        obs, _, _, _, info = env.step(np.array([a], np.float32))
    return {k: (np.array(v) if k == "obs" else np.array(v).reshape(-1, 1)) for k, v in log.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--z", default="resultats/agent_z_ppo")
    p.add_argument("--lacet", default=None, help="dossier de l'agent de lacet (par défaut : le plus récent)")
    p.add_argument("--out", default="../simulink_RL/zpsi")
    a = p.parse_args()
    lacet = Path(a.lacet) if a.lacet else sorted((HERE / "resultats").glob("agent_psi_ppo*"))[-1]
    out = (HERE / a.out).resolve()
    agent_z = PPO.load(HERE / a.z / "best_model", device="cpu")
    agent_psi = PPO.load(HERE / lacet / "best_model", device="cpu")

    savemat(out / "policy_z.mat", _policy_matrices(agent_z))
    savemat(out / "policy_psi.mat", _policy_matrices(agent_psi))
    tz = vol(agent_z, DroneEnv(z_ref_fixe=MISSION_AVANCEE_Z), ("t", "z", "vz", "az_prev", "z_ref", "u_prev"))
    tz["u"] = tz.pop("a")
    savemat(out / "traj_z.mat", tz)
    savemat(out / "traj_psi.mat", vol(agent_psi, YawEnv(sauts_fixes=MISSION_SCAN), ("t", "psi", "wz", "wdz_prev", "psi_ref", "a_prev")))
    print(f"Agents {a.z} (z) et {lacet.name} (lacet) exportés dans {out} : policy_z.mat, policy_psi.mat, traj_z.mat, traj_psi.mat")


if __name__ == "__main__":
    main()
