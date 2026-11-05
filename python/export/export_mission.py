"""
Export de la mission finale vers EI_RL/simulink_RL/mission/ (contrôleur rl_controller_mission.m).

    python python/export/export_z.py

Fichiers écrits :
  policy_z.mat, policy_psi.mat, policy_x.mat : les trois agents retenus (format policy.mat, fonction _policy_matrices
      du fichier fourni export_policy.py, utilisée sans le modifier) ;
  lq_attitude.mat : K du LQ d'attitude rapide (theta et phi) ;
  voiture.mat : table de la voiture (t, x, y, vx, vy, cap, v, kappa), pas de 0,05 s, lue par le contrôleur ;
  traj_mission_cap.mat : vol Python SANS bruit de la mission, configuration finale (CONFIG_FINALE : cap vers la
      voiture, biais de l'agent x corrigé) : v (entrées du contrôleur, N x 19) et
      y (sorties, N x 16). EINT_RL_eval_mission.m rejoue chaque ligne de v dans rl_controller_mission.m et compare y.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import EI, SIMULINK_RL  # noqa: E402
from env_x import THETA_MAX, charger_boucle_interne  # noqa: E402
import sim_mission as M  # noqa: E402
import voiture as V  # noqa: E402

T_FIN = 60.0


def main():
    out = SIMULINK_RL / "mission"; out.mkdir(parents=True, exist_ok=True)
    P = M.charger_politiques()
    for k in ("z", "psi", "x"):
        savemat(out / f"policy_{k}.mat", P[k])
    savemat(out / "lq_attitude.mat", dict(K_ATT=charger_boucle_interne()["K"].reshape(1, -1), THETA_MAX=THETA_MAX))
    tab = V.trajectoire(t_fin=T_FIN + 1)
    savemat(out / "voiture.mat", {k: tab[k].reshape(-1, 1) for k in ("t", "x", "y", "vx", "vy", "cap", "v", "kappa")})
    L = M.vol(P, tab, "rl", 0.0, t_fin=T_FIN, **M.CONFIG_FINALE)
    savemat(out / "traj_mission_cap.mat", dict(v=L["v"], y=L["y"], vrai=L["vrai"]))
    d = M.bilan_complet(L, tab)
    print(f"Exporté dans {out} : policy_z/psi/x.mat, lq_attitude.mat, voiture.mat, traj_mission_cap.mat ({len(L['v'])} pas, configuration finale {M.CONFIG_FINALE})")
    print(f"Vol Python : posé à {d['t_posé']} s, écart {d['ecart_contact_m']:.3f} m, "
          f"vitesse relative {d['vitesse_relative_horiz_ms']:.3f} m/s")


if __name__ == "__main__":
    main()
