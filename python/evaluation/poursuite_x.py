"""
Évaluation de la poursuite d'une voiture rapide (un axe) : trois contrôleurs sur les mêmes scénarios.

    original        agent x de la mission (resultats/agent_x_ppo_saturation), observation inchangée (vitesse absolue)
    original + ff   même agent, mais vitesse RELATIVE à la référence et anticipation de son accélération en inclinaison :
                        theta_ref = THETA_MAX a + clip(a_ref / g_x, +-0,15)     (corrections analytiques, sans réentraîner)
    réentraîné      agent de train_ppo_x_poursuite.py (observation de env_x_poursuite.py)

Critères, ceux du superviseur de sim_mission.py (phase 3 -> 4) : |écart à la voiture| < 5 cm et |vitesse relative| < 10 cm/s
pendant 40 pas consécutifs (2 s). Pour chaque vol : instant de pose, dépassement (de combien le drone passe AU-DELÀ de la
voiture avant de s'y stabiliser) et écart de traînage sur les 5 dernières secondes.

    python python/evaluation/poursuite_x.py [--agent chemin_du_nouvel_agent]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import PYTHON  # noqa: E402  (rend importables les sous-dossiers de python/)
from env_z import TS  # noqa: E402
from env_x import N_STEPS, THETA_MAX, I_MAX  # noqa: E402
from env_x_poursuite import (PoursuiteEnv, V_CAR_MAX, observation_poursuite, profil_voiture, scenario,  # noqa: E402
                             scenario_aleatoire)

SIGMA_G01 = 0.0045
AGENT_ORIGINAL = "resultats/agent_x_ppo_saturation"
TOL_P, TOL_V, N_ALIGNE = 0.05, 0.10, 40          # = sim_mission.py
FF_MAX = 0.15                                    # borne de l'inclinaison d'anticipation [rad]
N_ALEA = 100


# ------------------------------------------------------------------ scénarios
def _const(v, D0, v_drone0=0.0, tau=1.0, ev=()):
    vc, xc = profil_voiture(None, V_CAR_MAX, evenements=list(ev), v0=v)
    return scenario(vc, xc, D0, tau, v_drone0)


SCENARIOS = {
    "voiture à 2 m/s, départ à l'arrêt": lambda: _const(2.0, -3.0),
    "voiture à 2 m/s, drone lancé à 2,5 m/s": lambda: _const(2.0, -4.0, 2.5),
    "voiture arrêtée, drone lancé à 2 m/s": lambda: _const(0.0, -4.0, 2.0),
    "voiture qui freine de 2 à 0 m/s": lambda: _const(2.0, -2.0, ev=((5.0, 0.0),)),
    "voiture qui accélère de 0 à 2 m/s": lambda: _const(0.0, -1.0, ev=((3.0, 2.0),)),
}


def scenarios_aleatoires(n=N_ALEA):
    rng = np.random.default_rng(2026)              # graine fixe : toujours les mêmes vols
    return [scenario_aleatoire(rng, V_CAR_MAX) for _ in range(n)]


# ------------------------------------------------------------------ contrôleurs : m -> (theta_ref, a)
class Controleur:
    def reset(self): ...


class AgentNouveau(Controleur):
    def __init__(self, model, g_x):
        self.model, self.g_x = model, g_x

    def __call__(self, m, a_prev):
        a = float(np.clip(np.asarray(self.model.predict(observation_poursuite(m, self.g_x), deterministic=True)[0]).reshape(-1)[0], -1, 1))
        return THETA_MAX * a, a


class AgentOriginal(Controleur):
    """Agent de env_x.py. relatif=False, ff=False : observation d'origine ; True/True : corrections analytiques."""

    def __init__(self, model, g_x, corrige):
        self.model, self.g_x, self.corrige = model, g_x, corrige

    def __call__(self, m, a_prev):
        ff = float(np.clip(m["a_ref"] / self.g_x, -FF_MAX, FF_MAX)) if self.corrige else 0.0
        vr = (m["vx"] - m["v_ref"]) if self.corrige else m["vx"]
        o = np.array([(m["x_ref"] - m["x"]) / 2.0, vr, (m["theta"] - ff) / THETA_MAX, m["wy"], a_prev, m["int_e"] / I_MAX,
                      m["v_ref"] - m["vx"]], dtype=np.float32)
        a = float(np.clip(np.asarray(self.model.predict(o, deterministic=True)[0]).reshape(-1)[0], -1, 1))
        return THETA_MAX * a + ff, a


# ------------------------------------------------------------------ un vol
def vol(ctl, sc, sigma=0.0, graine=0):
    """Un vol de 20 s sur le scénario sc. Trajectoire (état VRAI) et indicateurs."""
    env = PoursuiteEnv(scenario_fixe=sc, sigma_x=sigma, sigma_vx=sigma, sigma_theta=sigma, sigma_wy=sigma)
    obs, info = env.reset(seed=graine)
    T = {k: [] for k in ("t", "x", "vx", "x_car", "v_car", "x_ref", "theta_ref")}
    for _ in range(N_STEPS):
        theta_ref, a = ctl(info["m"], env.a_prev)
        obs, r, _, _, info = env.avance(theta_ref, a)[0:5]
        k = env.k
        for key, v in zip(T, (k * TS, env.boucle.x, env.boucle.vx, sc["x_car"][k], sc["v_car"][k], sc["x_ref"][k], theta_ref)):
            T[key].append(v)
    T = {k: np.asarray(v) for k, v in T.items()}
    return T, indicateurs(T, sc)


def indicateurs(T, sc):
    gap, dv = T["x"] - T["x_car"], T["vx"] - T["v_car"]
    ok = (np.abs(gap) < TOL_P) & (np.abs(dv) < TOL_V)
    run, k_pose = 0, None
    for k, o in enumerate(ok):
        run = run + 1 if o else 0
        if run >= N_ALIGNE:
            k_pose = k - N_ALIGNE + 1                         # début de la fenêtre qui a déclenché la descente
            break
    # dépassement : de combien le drone passe au-delà de la voiture, du côté opposé à son départ
    depasse = float(max(0.0, np.max(-np.sign(sc["D0"]) * gap))) if abs(sc["D0"]) > 0.5 else float("nan")
    return dict(pose=k_pose is not None, t_pose=float(T["t"][k_pose]) if k_pose is not None else float("nan"),
                depassement=depasse, e_fin=float(np.mean(np.abs(gap[-100:]))), dv_fin=float(np.mean(np.abs(dv[-100:]))),
                gap_max=float(np.max(np.abs(gap))))


def resume(rows):
    pose = np.array([r["pose"] for r in rows])
    t = np.array([r["t_pose"] for r in rows]); d = np.array([r["depassement"] for r in rows])
    return dict(taux_pose=float(pose.mean()), t_pose_med=float(np.nanmedian(t)) if pose.any() else float("nan"),
                depassement_moy=float(np.nanmean(d)), depassement_p90=float(np.nanpercentile(d, 90)),
                depassement_max=float(np.nanmax(d)), e_fin_moy=float(np.mean([r["e_fin"] for r in rows])))


def evaluer(controleurs, sigma=SIGMA_G01, n=N_ALEA):
    """controleurs : {nom: Controleur}. Renvoie {nom: {'scenarios': {...}, 'aleatoire': résumé}} et les trajectoires."""
    res, traj = {}, {}
    alea = scenarios_aleatoires(n)
    for nom, ctl in controleurs.items():
        res[nom], traj[nom] = {"scenarios": {}}, {}
        for sn, f in SCENARIOS.items():
            T, ind = vol(ctl, f(), sigma)
            res[nom]["scenarios"][sn], traj[nom][sn] = ind, T
        res[nom]["aleatoire"] = resume([vol(ctl, sc, sigma, graine=i)[1] for i, sc in enumerate(alea)])
    return res, traj


def charger_controleurs(chemin_nouveau):
    from stable_baselines3 import PPO
    g_x = PoursuiteEnv().g_x
    ancien = PPO.load(PYTHON / AGENT_ORIGINAL / "best_model", device="cpu")
    c = {"original": AgentOriginal(ancien, g_x, False), "original + corrections": AgentOriginal(ancien, g_x, True)}
    if chemin_nouveau:
        c["réentraîné"] = AgentNouveau(PPO.load(Path(chemin_nouveau) / "best_model", device="cpu"), g_x)
    return c


def afficher(res):
    print(f"{'':<42}" + "".join(f"{n:>26}" for n in res))
    print(f"{'':<42}" + "".join(f"{'pose / dépass. [cm] / e_fin [cm]':>26}" for _ in res))
    for sn in SCENARIOS:
        print(f"{sn:<42}" + "".join(
            f"{('oui' if r['scenarios'][sn]['pose'] else 'NON'):>6} {100 * r['scenarios'][sn]['depassement']:>8.1f} {100 * r['scenarios'][sn]['e_fin']:>8.1f}   "
            for r in res.values()))
    print(f"{'100 vols aléatoires (bruit g=0,1) :':<42}")
    for lab, key, f in (("  taux de pose", "taux_pose", "{:.0%}"), ("  instant de pose médian [s]", "t_pose_med", "{:.1f}"),
                        ("  dépassement moyen [cm]", "depassement_moy", "{:.1f}"), ("  dépassement p90 [cm]", "depassement_p90", "{:.1f}"),
                        ("  dépassement max [cm]", "depassement_max", "{:.1f}")):
        k = 100 if "cm" in lab else 1
        print(f"{lab:<42}" + "".join(f"{f.format(k * r['aleatoire'][key]):>26}" for r in res.values()))


def figure(traj, chemin):
    noms = list(traj)
    fig, ax = plt.subplots(len(SCENARIOS), 2, figsize=(13, 2.6 * len(SCENARIOS)), sharex=True)
    for i, sn in enumerate(SCENARIOS):
        for nom in noms:
            T = traj[nom][sn]
            ax[i, 0].plot(T["t"], T["x"] - T["x_car"], label=nom); ax[i, 1].plot(T["t"], T["vx"] - T["v_car"], label=nom)
        ax[i, 0].axhspan(-TOL_P, TOL_P, color="g", alpha=.15); ax[i, 1].axhspan(-TOL_V, TOL_V, color="g", alpha=.15)
        ax[i, 0].set_ylim(-4.5, 2.5); ax[i, 1].set_ylim(-3, 3)
        ax[i, 0].set_ylabel("écart drone - voiture [m]"); ax[i, 1].set_ylabel("vitesse relative [m/s]")
        ax[i, 0].set_title(sn, fontsize=9, loc="left"); ax[i, 0].grid(alpha=.3); ax[i, 1].grid(alpha=.3)
    ax[0, 0].legend(fontsize=8); ax[-1, 0].set_xlabel("t [s]"); ax[-1, 1].set_xlabel("t [s]")
    fig.tight_layout(); fig.savefig(chemin, dpi=110)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default=None, help="dossier du nouvel agent (contient best_model.zip)")
    a = p.parse_args()
    res, traj = evaluer(charger_controleurs(a.agent))
    afficher(res)
    sortie = Path(a.agent) if a.agent else PYTHON / "resultats"
    figure(traj, sortie / "poursuite_comparaison.png")
    json.dump(res, open(sortie / "poursuite_comparaison.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
