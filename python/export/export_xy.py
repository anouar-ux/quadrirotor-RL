"""
Étape C3 : export pour le vol Simulink de la position (x, y), vers EI_RL/simulink_RL/xy/ (contrôleur controleur_xy.m).

    python python/export/export_xy.py

Fichiers écrits :
  policy_x.mat (agent x retenu, utilisé pour LES DEUX axes), policy_z.mat, policy_psi.mat : format de policy.mat
      (fonction _policy_matrices du fichier fourni export_policy.py, utilisée sans le modifier) ;
  lq_attitude.mat : K du LQ d'attitude rapide (le même pour theta et phi) ;
  traj_xy.mat : vol Python SANS bruit de la mission C3 (sim_xy.vol, journal complet : mesures, observations des deux
      axes, actions, commandes u_theta / u_phi du LQ) ; EINT_RL_eval_xy.m le rejoue dans controleur_xy.m.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import EI, RESULTATS, RL_INTERFACE, SIMULINK_RL  # noqa: E402
from env_x import THETA_MAX, charger_boucle_interne  # noqa: E402
from sim_xy import AGENT_X, N_PAS, consignes, politique_agent, vol  # noqa: E402

sys.path.insert(0, str(RL_INTERFACE))
from export_policy import _policy_matrices  # noqa: E402  (fichier fourni, non modifié)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--x", default=AGENT_X)
    p.add_argument("--z", default="resultats/agent_z_ppo")
    p.add_argument("--lacet", default="resultats/agent_psi_ppo")
    p.add_argument("--out", default=str(SIMULINK_RL / "xy"))
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rac = RESULTATS.parent
    for nom, chemin in (("x", a.x), ("z", a.z), ("psi", a.lacet)):
        savemat(out / f"policy_{nom}.mat", _policy_matrices(PPO.load(rac / chemin / "best_model", device="cpu")))
    K = charger_boucle_interne()["K"]
    savemat(out / "lq_attitude.mat", dict(K_ATT=K.reshape(1, -1), THETA_MAX=THETA_MAX))

    T = vol(politique_agent(a.x), journal=True)
    J = T.pop("journal")
    R = np.array([consignes(t)[0] for t in J["t"]])                 # x_ref, y_ref, vx_ref, vy_ref
    traj = {k: (v if v.ndim == 2 else v.reshape(-1, 1)) for k, v in J.items()}
    traj.update(refs=R, psi_ref=np.array([consignes(t)[1] for t in J["t"]]).reshape(-1, 1),
                x_apres=np.r_[T["x"][1:], np.nan].reshape(-1, 1), y_apres=np.r_[T["y"][1:], np.nan].reshape(-1, 1),
                x_vrai=T["x"].reshape(-1, 1), y_vrai=T["y"].reshape(-1, 1))
    savemat(out / "traj_xy.mat", traj)
    print(f"Exporté dans {out} : policy_x.mat ({a.x}), policy_z.mat, policy_psi.mat, lq_attitude.mat, traj_xy.mat "
          f"({N_PAS} pas)")
    print(f"Vol Python : position finale ({T['x'][-1]:.3f} ; {T['y'][-1]:.3f}) m, consigne ({R[-1, 0]:.3f} ; {R[-1, 1]:.3f})")


if __name__ == "__main__":
    main()
