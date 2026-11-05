"""
Étape C2.4 : export des agents pour le vol Simulink de la position x, vers EI_RL/simulink_RL/x/ (contrôleur controleur_x.m).

    python python/export/export_x.py                                  # agent x retenu (saturations) + agents z et lacet retenus
    python python/export/export_x.py --x resultats/agent_x_ppo   # un autre agent x

Fichiers écrits :
  policy_x.mat, policy_z.mat, policy_psi.mat : les acteurs, au format de policy.mat (fonction _policy_matrices du
      fichier fourni export_policy.py, utilisée sans le modifier).
  lq_attitude.mat : gain K (9 termes) du LQ d'attitude RAPIDE de la boucle interne (env_x.py : Q_ANG, Q_W, R_ATT),
      le même pour le tangage et le roulis.
  traj_x.mat : vol Python SANS bruit de l'agent x sur MISSION_X (mêmes consignes que controleur_x.m), avec les
      mesures, observations, actions ET la commande u_theta du LQ : EINT_RL_eval_x.m rejoue tout dans controleur_x.m
      pour vérifier qu'il calcule exactement la même chose, puis compare le vol Simulink à ce vol prédit.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import EI, RESULTATS, RL_INTERFACE, SIMULINK_RL  # noqa: E402  (rend importables les sous-dossiers de python/)
from env_x import THETA_MAX, XEnv, charger_boucle_interne, consigne_x  # noqa: E402

sys.path.insert(0, str(RL_INTERFACE))
from export_policy import _policy_matrices  # noqa: E402  (fichier fourni, non modifié)

# Mission x du vol Simulink (À RECOPIER À L'IDENTIQUE dans consigne_x() de controleur_x.m) :
#   saut de +2 m à 4 s, retour à 12 s, puis la « voiture » apparaît 5,5 m devant à 20 s et roule à 0,3 m/s.
MISSION_X = ((4.0, "p", 2.0), (12.0, "p", -2.0), (20.0, "p", 5.5), (20.0, "v", 0.3))
T_FIN = 40.0
N_PAS = int(round(T_FIN / 0.05))


def vol_python(agent):
    env = XEnv(evenements_fixes=MISSION_X)
    obs, info = env.reset(seed=0)
    B = env.boucle
    cles = ("t", "x", "vx", "theta", "wy", "a_prev", "x_ref")
    log = {k: [] for k in (*cles, "v_ref", "wdy", "wdy_prev", "U", "obs", "a", "u_theta")}
    for _ in range(N_PAS):
        m = info["m"]
        a = float(np.asarray(agent.predict(obs, deterministic=True)[0]).reshape(-1)[0])
        for k in cles:
            log[k].append(float(m[k]))
        log["v_ref"].append(consigne_x(m["t"], env.ev)[1])
        log["wdy"].append(B.wdy); log["wdy_prev"].append(B.wdy_prev); log["U"].append(B.U.copy())
        log["obs"].append(np.asarray(obs, np.float64)); log["a"].append(a)
        obs, _, _, _, info = env.step(np.array([a], np.float32))
        log["u_theta"].append(float(B.U[0]))        # commande du LQ appliquée pendant ce pas
    out = {k: (np.array(v) if k in ("obs", "U") else np.array(v).reshape(-1, 1)) for k, v in log.items()}
    out["x_apres"] = np.r_[out["x"][1:, 0], B.x].reshape(-1, 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--x", default="resultats/agent_x_ppo_saturation")
    p.add_argument("--z", default="resultats/agent_z_ppo")
    p.add_argument("--lacet", default="resultats/agent_psi_ppo")
    p.add_argument("--out", default=str(SIMULINK_RL / "x"))
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rac = RESULTATS.parent
    agent_x = PPO.load(rac / a.x / "best_model", device="cpu")
    savemat(out / "policy_x.mat", _policy_matrices(agent_x))
    savemat(out / "policy_z.mat", _policy_matrices(PPO.load(rac / a.z / "best_model", device="cpu")))
    savemat(out / "policy_psi.mat", _policy_matrices(PPO.load(rac / a.lacet / "best_model", device="cpu")))
    K = charger_boucle_interne()["K"]
    savemat(out / "lq_attitude.mat", dict(K_ATT=K.reshape(1, -1), THETA_MAX=THETA_MAX))
    T = vol_python(agent_x)
    savemat(out / "traj_x.mat", T)
    print(f"Exporté dans {out} : policy_x.mat ({a.x}), policy_z.mat, policy_psi.mat, lq_attitude.mat, traj_x.mat")
    print(f"K_ATT = {np.round(K, 5)}")
    print(f"Vol Python : x final {T['x_apres'][-1, 0]:.3f} m, consigne {T['x_ref'][-1, 0]:.3f} m, "
          f"écart max après 30 s {np.max(np.abs(T['x_ref'][600:, 0] - T['x'][600:, 0])):.3f} m")


if __name__ == "__main__":
    main()
