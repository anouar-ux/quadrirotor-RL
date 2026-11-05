"""
Évaluation des contrôleurs de position x sur l'environnement XEnv (étape C2.2), et RÉFÉRENCE PD.

Un contrôleur est une fonction politique(obs, m, v_ref) -> a dans [-1, 1] (theta_ref = THETA_MAX * a) :
  - PD de la cascade LQI (cascade LQ) : theta_ref = Kp (x_ref - x) - Kd vx                  (sans anticipation)
  - PD avec anticipation          : theta_ref = Kp (x_ref - x) + Kd (v_ref - vx)           (version mission)
  - plus tard l'agent PPO          : a = model.predict(obs)
Scénarios fixes (les mêmes serviront pour l'agent PPO à l'étape C2.3) :
  - saut de 2 m, cible qui roule à 0,3 m/s (voiture), cible à 1 m/s, et 20 missions aléatoires (graines 0 à 19),
  - sans bruit et avec un bruit de mesure équivalent à g_noise = 0,1 (écart-type 0,0045 sur chaque mesure).
Lancer : python python/evaluation/reference_pd_x.py   ->  resultats/reference_pd_x/ (resultats.json, figure)
"""
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import PYTHON  # noqa: E402  (rend importables les sous-dossiers de python/)
HERE = PYTHON                # racine python/ : modeles/, resultats/ ; HERE.parent = EI_RL/
from env_z import TS  # noqa: E402
from env_x import N_STEPS, THETA_MAX, XEnv, consigne_x, mission_aleatoire  # noqa: E402

SORTIE = HERE / "resultats" / "reference_pd_x"
SIGMA_G01 = 0.0045            # écart-type du bruit de mesure de UAV_model à g_noise = 0,1 (sqrt(1e-4 / 0,05) x 0,1)

SCENARIOS = {
    "saut_2m": ((1.0, "p", 2.0),),
    "saut_5m": ((1.0, "p", 5.0),),          # demande ~4,6 m/s² : AU-DELÀ de theta_max (~2 m/s²) -> saturation
    "voiture_0.3": ((1.0, "v", 0.3),),
    "cible_1.0": ((1.0, "v", 1.0),),
}
N_ALEA = 20


def missions_alea():
    rng = np.random.default_rng(12345)          # graine fixe : toujours les mêmes 20 missions
    return [mission_aleatoire(rng) for _ in range(N_ALEA)]


# ------------------------------------------------------------------ contrôleurs PD
def pd_cascade_lqi(Kp=0.23, Kd=0.28):
    """Cascade LQI (commande_LQ/gains_LQI.m) : pas de vitesse de consigne."""
    return lambda obs, m, v_ref: float(np.clip(Kp * (m["x_ref"] - m["x"]) - Kd * m["vx"], -THETA_MAX, THETA_MAX)) / THETA_MAX


def pd_anticipation(Kp=0.23, Kd=0.28):
    """PD avec la vitesse de consigne (version mission d'atterrissage) : plus d'erreur de traînage."""
    return lambda obs, m, v_ref: float(np.clip(Kp * (m["x_ref"] - m["x"]) + Kd * (v_ref - m["vx"]), -THETA_MAX, THETA_MAX)) / THETA_MAX


# ------------------------------------------------------------------ évaluation
def vol(politique, evenements, sigma=0.0, graine=0):
    """Un vol de 20 s. Renvoie la trajectoire (état VRAI) et les indicateurs."""
    env = XEnv(evenements_fixes=evenements, sigma_x=sigma, sigma_vx=sigma, sigma_theta=sigma, sigma_wy=sigma)
    obs, info = env.reset(seed=graine)
    T = {k: [] for k in ("t", "x", "vx", "theta", "x_ref", "v_ref", "a", "r")}
    R = 0.0
    for _ in range(N_STEPS):
        m = info["m"]
        v_ref = consigne_x(m["t"], env.ev)[1]
        a = float(np.clip(politique(obs, m, v_ref), -1.0, 1.0))
        obs, r, _, _, info = env.step([a])
        R += r
        B = env.boucle
        xr, vr = consigne_x(env.k * TS, env.ev)
        for k, v in zip(T, (env.k * TS, B.x, B.vx, B.theta, xr, vr, a, r)):
            T[k].append(v)
    T = {k: np.asarray(v) for k, v in T.items()}
    e = T["x_ref"] - T["x"]
    da = np.abs(np.diff(np.r_[0.0, T["a"]]))
    ind = dict(rendement=float(R), rmse=float(np.sqrt(np.mean(e ** 2))), e_max=float(np.max(np.abs(e))),
               e_fin=float(np.mean(np.abs(e[-60:]))),                 # écart moyen sur les 3 dernières secondes
               agitation=float(np.mean(da)), saturation=float(np.mean(np.abs(T["a"]) > 0.99)))
    return T, ind


def indicateurs_saut(T, dx=2.0, t0=1.0):
    """Montée 10-90 %, dépassement et temps de réponse à 5 cm, sur la position (pas la consigne lissée)."""
    t, x = T["t"], T["x"]
    s = t >= t0
    k1, k9 = np.argmax(x >= 0.1 * dx), np.argmax(x >= 0.9 * dx)
    hors = np.where(np.abs(x - dx) > 0.05)[0]
    return dict(montee=float(t[k9] - t[k1]) if x.max() >= 0.9 * dx else np.nan,
                depassement=float(100 * (x[s].max() - dx) / dx),
                reponse=float(t[hors[-1] + 1] - t0) if len(hors) and hors[-1] + 1 < len(t) else np.nan)


def evaluer(politique, sigma=0.0):
    """Tous les scénarios. Renvoie {scénario: indicateurs} + la moyenne sur les missions aléatoires."""
    res, traj = {}, {}
    for nom, ev in SCENARIOS.items():
        T, ind = vol(politique, ev, sigma)
        if nom.startswith("saut_"):
            ind.update(indicateurs_saut(T, dx=ev[0][2]))
        res[nom], traj[nom] = ind, T
    alea = [vol(politique, ev, sigma, graine=i)[1] for i, ev in enumerate(missions_alea())]
    res["aleatoire_moyenne"] = {k: float(np.mean([a[k] for a in alea])) for k in alea[0]}
    return res, traj


def recherche_gains(sigma=0.0):
    """Meilleur PD avec anticipation sur les 20 missions aléatoires (rendement moyen) : référence la plus forte."""
    meilleur = None
    for Kp in (0.1, 0.15, 0.23, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0):
        for Kd in (0.15, 0.2, 0.28, 0.35, 0.45, 0.6, 0.8):
            pol = pd_anticipation(Kp, Kd)
            R = np.mean([vol(pol, ev, sigma, graine=i)[1]["rendement"] for i, ev in enumerate(missions_alea())])
            if meilleur is None or R > meilleur[0]:
                meilleur = (R, Kp, Kd)
    return meilleur


if __name__ == "__main__":
    SORTIE.mkdir(parents=True, exist_ok=True)
    R_opt, Kp_o, Kd_o = recherche_gains()
    print(f"Recherche de gains (PD avec anticipation, 20 missions, sans bruit) : Kp = {Kp_o}, Kd = {Kd_o}, "
          f"rendement moyen {R_opt:.1f}")
    ctrl = {"PD cascade LQI (0.23 / 0.28)": pd_cascade_lqi(),
            "PD + anticipation (0.23 / 0.28)": pd_anticipation(),
            f"PD + anticipation optimisé ({Kp_o} / {Kd_o})": pd_anticipation(Kp_o, Kd_o)}
    tout, trajs = {}, {}
    for nom, pol in ctrl.items():
        for sig in (0.0, SIGMA_G01):
            res, traj = evaluer(pol, sig)
            tout[f"{nom} | bruit {sig}"] = res
            if sig == 0.0:
                trajs[nom] = traj
    # ---- affichage
    for cle, res in tout.items():
        print(f"\n=== {cle}")
        for n in ("saut_2m", "saut_5m"):
            s = res[n]
            print(f"  {n:13s} : montée {s['montee']:.2f} s, dépassement {s['depassement']:.1f} %, réponse (5 cm) "
                  f"{s['reponse']:.2f} s, RMSE {s['rmse']:.3f} m, saturation {100 * s['saturation']:.0f} %, "
                  f"rendement {s['rendement']:.1f}")
        for n in ("voiture_0.3", "cible_1.0"):
            s = res[n]
            print(f"  {n:13s} : écart final {100 * s['e_fin']:.1f} cm, écart max {s['e_max']:.3f} m, RMSE {s['rmse']:.3f} m, "
                  f"rendement {s['rendement']:.1f}")
        s = res["aleatoire_moyenne"]
        print(f"  20 missions   : rendement {s['rendement']:.1f}, RMSE {s['rmse']:.3f} m, écart max {s['e_max']:.3f} m, "
              f"agitation {s['agitation']:.4f}, saturation {100 * s['saturation']:.1f} %")
    with open(SORTIE / "resultats.json", "w", encoding="utf-8") as f:
        json.dump(dict(gains_optimaux=dict(Kp=Kp_o, Kd=Kd_o, rendement=R_opt), resultats=tout), f, indent=2,
                  ensure_ascii=False)
    # ---- figure : saut et voiture
    fig, ax = plt.subplots(3, 3, figsize=(16, 9), sharex="col")
    for j, sc in enumerate(("saut_2m", "saut_5m", "voiture_0.3")):
        for nom, traj in trajs.items():
            T = traj[sc]
            ax[0, j].plot(T["t"], T["x"], label=nom)
            ax[1, j].plot(T["t"], T["x_ref"] - T["x"])
            ax[2, j].plot(T["t"], THETA_MAX * T["a"])
        T = next(iter(trajs.values()))[sc]
        ax[0, j].plot(T["t"], T["x_ref"], "k--", label="consigne lissée")
        ax[0, j].set_title({"saut_2m": "Saut de 2 m", "saut_5m": "Saut de 5 m (θ saturé)",
                            "voiture_0.3": "Cible qui roule à 0,3 m/s (voiture)"}[sc])
        ax[0, j].set_ylabel("x [m]"); ax[1, j].set_ylabel("erreur x_ref - x [m]"); ax[2, j].set_ylabel("θ_ref [rad]")
        ax[2, j].set_xlabel("t [s]")
        for a in ax[:, j]:
            a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Référence PD sur l'environnement x (boucle interne LQ d'attitude), sans bruit")
    fig.tight_layout()
    fig.savefig(SORTIE / "reference_pd.png", dpi=120)
    print(f"\nRésultats : {SORTIE}")
