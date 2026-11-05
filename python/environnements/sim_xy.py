"""
Étape C3 : position (x, y) avec l'agent x RÉUTILISÉ sur les deux axes, et rotation des erreurs par psi.

    python python/environnements/sim_xy.py            ->  resultats/essais_xy/ (resultats_xy.json, sim_xy.png)

Pourquoi un seul agent suffit (identification C1, modeles/attitude/results.json) :
  - attitude : ARX de phi = ARX de theta (gain 171,76 / 171,78 rad/s² par unité, couplage < 2 %) -> même LQ (K_ATT) ;
  - translation : ax = +9,91 theta, ay = -9,91 phi.  Avec theta' = -phi, l'axe y du drone est un axe x identique.
Rotation par psi : theta et phi inclinent le drone selon SES axes, la consigne et les mesures (x, y, vx, vy) sont dans
le repère du SOL. On projette tout dans le repère du drone (même convention que le bloc rotation_psi de la mission LQ) :
    e_avant =  cos(psi) e_x + sin(psi) e_y          e_cote = -sin(psi) e_x + cos(psi) e_y
puis :  agent(obs avant) -> theta_ref = 0,2 a_x  ;  agent(obs côté, avec theta' = -phi, w' = -wx) -> phi_ref = -0,2 a_y.
Les intégrales d'erreur sont tenues dans le repère du drone (elles suivent l'axe sur lequel elles agissent).

Modèle simulé (tout identifié) : deux boucles d'attitude ARX + LQ (comme env_x.BoucleTangage), et une translation dans
le repère du SOL :  a_sol = R(psi) [G_X theta ; G_Y phi] + frottement.  psi suit sa consigne (scan) exactement : son
agent est validé séparément (étape B2), Simulink le fera voler pour de vrai.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import PYTHON, RESULTATS  # noqa: E402  (rend importables les sous-dossiers de python/)
from env_z import TS  # noqa: E402
from env_x import (E_INT, I_MAX, THETA_MAX, U_MAX_ATT, charger_boucle_interne, consigne_x)  # noqa: E402
from env_psi import MISSION_SCAN, consigne_lacet  # noqa: E402

AGENT_X = "resultats/agent_x_ppo_saturation"     # agent x retenu (C2.3, entraîné aux saturations)
SORTIE = RESULTATS / "essais_xy"
SIGMA_G01 = 0.0045                             # écart-type du bruit de mesure à g_noise = 0,1
KP_PD, KD_PD = 0.3, 0.45                       # meilleure référence PD avec anticipation (C2.2)

# Mission C3 (45 s), À RECOPIER À L'IDENTIQUE dans controleur_xy.m :
#   décollage à 1 s ; scan 360° entre 5 et 13 s en tenant (0, 0) ; à 16 s, déplacement en DIAGONALE vers (2 ; 1,5) ;
#   à 24 s, la « voiture » apparaît en (5,5 ; 0), soit 3,5 m devant et 1,5 m sur le côté, et roule selon x à 0,3 m/s.
MISSION_XY_X = ((16.0, "p", 2.0), (24.0, "p", 3.5), (24.0, "v", 0.3))
MISSION_XY_Y = ((16.0, "p", 1.5), (24.0, "p", -1.5))
T_FIN = 45.0
N_PAS = int(round(T_FIN / TS))


def rot_vers_drone(psi, vx, vy):
    """Vecteur du repère sol -> repère du drone (avant, côté)."""
    c, s = np.cos(psi), np.sin(psi)
    return c * vx + s * vy, -s * vx + c * vy


class Attitude:
    """Une boucle d'attitude : LQ rapide (K_ATT) + ARX(2,6). Même calcul que env_x.BoucleTangage, sans la translation."""

    def __init__(self, p):
        self.p = p
        self.ang = self.w = self.wd = self.wd_prev = 0.0
        self.U = np.zeros(5)

    def commande(self, ang_mes, w_mes, wd_mes, wd_prev_mes, ang_ref):
        u = -self.p["K"] @ np.r_[ang_mes - ang_ref, w_mes, wd_mes, wd_prev_mes, self.U]
        return float(np.clip(u, -U_MAX_ATT, U_MAX_ATT))

    def avance(self, u):
        p = self.p
        wd_next = p["a"][0] * self.wd + p["a"][1] * self.wd_prev + p["b"][0] * u + p["b"][1:] @ self.U
        w_next = self.w + TS / 2 * (self.wd + wd_next)
        self.ang += TS / 2 * (self.w + w_next)
        self.U = np.r_[u, self.U[:4]]
        self.wd_prev, self.wd, self.w = self.wd, wd_next, w_next


class ControleurXY:
    """Contrôleur de position (x, y) : rotation par psi, puis agent x (ou PD) sur chaque axe du drone.
    politique(obs) -> a dans [-1, 1]. Même calcul dans controleur_xy.m."""

    def __init__(self, politique, rotation=True):
        self.pol = politique
        self.rotation = rotation
        self.int_av = self.int_co = 0.0
        self.a_av = self.a_co = 0.0

    @staticmethod
    def _integre(i, e):
        if abs(e) < E_INT:
            i += e * TS
        return float(np.clip(i, -I_MAX, I_MAX))

    def __call__(self, m, ref):
        """m : mesures (x, y, vx, vy, theta, phi, wx, wy, psi) ; ref : (x_ref, y_ref, vx_ref, vy_ref).
        Renvoie theta_ref, phi_ref et les deux observations."""
        psi = m["psi"] if self.rotation else 0.0
        e_av, e_co = rot_vers_drone(psi, ref[0] - m["x"], ref[1] - m["y"])
        v_av, v_co = rot_vers_drone(psi, m["vx"], m["vy"])
        vr_av, vr_co = rot_vers_drone(psi, ref[2], ref[3])
        self.int_av = self._integre(self.int_av, e_av)
        self.int_co = self._integre(self.int_co, e_co)
        # mêmes 7 composantes que env_x.observation ; axe côté en miroir : theta' = -phi, w' = -wx
        o_av = np.array([e_av / 2, v_av, m["theta"] / THETA_MAX, m["wy"], self.a_av, self.int_av / I_MAX, vr_av - v_av],
                        np.float32)
        o_co = np.array([e_co / 2, v_co, -m["phi"] / THETA_MAX, -m["wx"], self.a_co, self.int_co / I_MAX, vr_co - v_co],
                        np.float32)
        self.a_av, self.a_co = self.pol(o_av), self.pol(o_co)
        return THETA_MAX * self.a_av, -THETA_MAX * self.a_co, o_av, o_co


def politique_agent(chemin=AGENT_X):
    from stable_baselines3 import PPO
    agent = PPO.load(PYTHON / chemin / "best_model", device="cpu")
    return lambda o: float(np.clip(np.asarray(agent.predict(o, deterministic=True)[0]).reshape(-1)[0], -1, 1))


def politique_pd(Kp=KP_PD, Kd=KD_PD):
    """PD avec anticipation, écrit sur l'observation : e = 2 o[0], erreur de vitesse = o[6]."""
    return lambda o: float(np.clip(Kp * 2 * o[0] + Kd * o[6], -THETA_MAX, THETA_MAX)) / THETA_MAX


def consignes(t, ev_x=MISSION_XY_X, ev_y=MISSION_XY_Y, scan=MISSION_SCAN):
    x_ref, vx_ref = consigne_x(t, ev_x)
    y_ref, vy_ref = consigne_x(t, ev_y)
    psi_ref = consigne_lacet(t, scan)[0] if scan else 0.0
    return (x_ref, y_ref, vx_ref, vy_ref), psi_ref


def vol(politique, rotation=True, sigma=0.0, graine=0, psi_fixe=None, ev_x=MISSION_XY_X, ev_y=MISSION_XY_Y,
        scan=MISSION_SCAN, n_pas=N_PAS, journal=False):
    """Vol 2-D. psi suit sa consigne (scan), ou psi_fixe (nombre, ou fonction de t). Renvoie la trajectoire (état VRAI)."""
    p = charger_boucle_interne()
    gx = p["g_x"]
    with open(PYTHON / "modeles" / "attitude" / "results.json", encoding="utf-8") as f:
        ty = json.load(f)["translation_y"]
    gy, fy, fx = float(ty["gain_angle"]), float(ty["frottement"]), p["frot_x"]
    th, ph = Attitude(p), Attitude(p)
    ctl = ControleurXY(politique, rotation)
    rng = np.random.default_rng(graine)
    b = (lambda: float(rng.normal(0.0, sigma))) if sigma > 0 else (lambda: 0.0)
    pos, vit, acc = np.zeros(2), np.zeros(2), np.zeros(2)
    nwd_prev = [0.0, 0.0]
    T = {k: [] for k in ("t", "x", "y", "vx", "vy", "theta", "phi", "psi", "x_ref", "y_ref", "theta_ref", "phi_ref")}
    J = {k: [] for k in ("t", "x", "y", "vx", "vy", "theta", "phi", "wx", "wy", "wdx", "wdy", "wdx_prev", "wdy_prev",
                         "psi", "Uth", "Uph", "obs_av", "obs_co", "a_av", "a_co", "u_theta", "u_phi")}
    for k in range(n_pas):
        t = k * TS
        ref, psi_ref = consignes(t, ev_x, ev_y, scan)
        psi = (psi_fixe(t) if callable(psi_fixe) else psi_fixe) if psi_fixe is not None else psi_ref
        n = [b() for _ in range(10)]
        m = dict(x=pos[0] + n[0], y=pos[1] + n[1], vx=vit[0] + n[2], vy=vit[1] + n[3], theta=th.ang + n[4],
                 phi=ph.ang + n[5], wy=th.w + n[6], wx=ph.w + n[7], psi=psi + b())
        theta_ref, phi_ref, o_av, o_co = ctl(m, ref)
        wdy_m, wdx_m = th.wd + n[8], ph.wd + n[9]
        u_th = th.commande(m["theta"], m["wy"], wdy_m, th.wd_prev + nwd_prev[0], theta_ref)
        u_ph = ph.commande(m["phi"], m["wx"], wdx_m, ph.wd_prev + nwd_prev[1], phi_ref)
        if journal:
            for c, v in (("t", t), ("x", m["x"]), ("y", m["y"]), ("vx", m["vx"]), ("vy", m["vy"]), ("theta", m["theta"]),
                         ("phi", m["phi"]), ("wx", m["wx"]), ("wy", m["wy"]), ("wdx", wdx_m), ("wdy", wdy_m),
                         ("wdx_prev", ph.wd_prev), ("wdy_prev", th.wd_prev), ("psi", m["psi"]), ("Uth", th.U.copy()),
                         ("Uph", ph.U.copy()), ("obs_av", o_av.astype(float)), ("obs_co", o_co.astype(float)),
                         ("a_av", ctl.a_av), ("a_co", ctl.a_co), ("u_theta", u_th), ("u_phi", u_ph)):
                J[c].append(v)
        nwd_prev = [n[8], n[9]]
        for c, v in (("t", t), ("x", pos[0]), ("y", pos[1]), ("vx", vit[0]), ("vy", vit[1]), ("theta", th.ang),
                     ("phi", ph.ang), ("psi", psi), ("x_ref", ref[0]), ("y_ref", ref[1]), ("theta_ref", theta_ref),
                     ("phi_ref", phi_ref)):
            T[c].append(v)
        th.avance(u_th); ph.avance(u_ph)
        # translation dans le repère du SOL (trapèzes, comme env_x)
        psi_next = ((psi_fixe(t + TS) if callable(psi_fixe) else psi_fixe) if psi_fixe is not None
                    else consignes(t + TS, ev_x, ev_y, scan)[1])
        c, s = np.cos(psi_next), np.sin(psi_next)
        a_b = np.array([gx * th.ang, gy * ph.ang])
        a_next = np.array([c * a_b[0] - s * a_b[1], s * a_b[0] + c * a_b[1]]) + np.array([fx, fy]) * vit
        v_next = vit + TS / 2 * (acc + a_next)
        pos = pos + TS / 2 * (vit + v_next)
        vit, acc = v_next, a_next
    T = {k: np.array(v) for k, v in T.items()}
    if journal:
        T["journal"] = {k: np.array(v) for k, v in J.items()}
    return T


def indicateurs(T):
    t = T["t"]
    e = np.hypot(T["x_ref"] - T["x"], T["y_ref"] - T["y"])
    i_scan = (t >= 5) & (t < 14)
    i_diag = (t >= 16) & (t < 24)
    i_voit = t >= 35
    # dépassement en x quand la voiture est rattrapée (le drone passe-t-il devant ?)
    i_rat = (t >= 24) & (t < 35)
    return dict(rmse=float(np.sqrt(np.mean(e[t >= 3] ** 2))),
                derive_scan=float(e[i_scan].max()),
                depass_diag=float(max(T["x"][i_diag].max() - 2.0, T["y"][i_diag].max() - 1.5, 0.0)),
                depass_voiture=float(max((T["x"][i_rat] - T["x_ref"][i_rat]).max(), 0.0)),
                ecart_voiture=float(e[i_voit].max()),
                theta_max=float(np.abs(T["theta"]).max()), phi_max=float(np.abs(T["phi"]).max()))


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)
    pol = {"agent PPO x": politique_agent(), "PD (0,3 / 0,45)": politique_pd()}
    R, V = {}, {}
    for nom, p in pol.items():
        for sig, gn in ((0.0, "g=0"), (SIGMA_G01, "g=0,1")):
            T = vol(p, sigma=sig, graine=1)
            V[(nom, gn)] = T
            R[f"{nom}, {gn}"] = indicateurs(T)
    # rôle de la rotation : cap tenu à 90° pendant toute la mission, avec et sans rotation
    for rot in (True, False):
        T = vol(pol["agent PPO x"], rotation=rot, psi_fixe=np.pi / 2, scan=None, sigma=SIGMA_G01, graine=1)
        V[("rotation", rot)] = T
        R[f"agent, psi = 90°, {'avec' if rot else 'SANS'} rotation, g=0,1"] = indicateurs(T)
    with open(SORTIE / "resultats_xy.json", "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False)
    print(f"{'cas':48s} {'RMSE':>7s} {'dérive scan':>11s} {'dép. diag':>9s} {'dép. voit.':>10s} {'écart voit.':>11s}")
    for k, r in R.items():
        print(f"{k:48s} {r['rmse']:7.3f} {r['derive_scan']:11.3f} {r['depass_diag']:9.3f} {r['depass_voiture']:10.3f} "
              f"{r['ecart_voiture']:11.3f}")

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(2, 2, 1)
    T0 = V[("agent PPO x", "g=0")]
    ax.plot(T0["x_ref"], T0["y_ref"], "k:", label="consigne")
    for (nom, gn), T in V.items():
        if nom in pol:
            ax.plot(T["x"], T["y"], "-" if gn == "g=0" else "--", lw=1.1, label=f"{nom}, {gn}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title("Trajectoire dans le plan (vue de dessus)")
    ax.axis("equal"); ax.grid(True); ax.legend(fontsize=8)
    for i, (c, lab) in enumerate((("x", "x [m]"), ("y", "y [m]"))):
        ax = fig.add_subplot(2, 2, 2 + i)
        ax.plot(T0["t"], T0[f"{c}_ref"], "k:", label="consigne")
        for (nom, gn), T in V.items():
            if nom in pol:
                ax.plot(T["t"], T[c], "-" if gn == "g=0" else "--", lw=1.1, label=f"{nom}, {gn}")
        ax.axvspan(5, 13, color="0.9", label="scan 360°")
        ax.set_ylabel(lab); ax.set_xlabel("t [s]"); ax.grid(True)
        if i == 0:
            ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 2, 4)
    for rot in (True, False):
        T = V[("rotation", rot)]
        ax.plot(T["t"], np.hypot(T["x_ref"] - T["x"], T["y_ref"] - T["y"]), label=f"{'avec' if rot else 'sans'} rotation")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1e3)
    ax.set_title("Cap tenu à 90° : erreur de position |e| (agent, g = 0,1)"); ax.set_xlabel("t [s]"); ax.set_ylabel("|e| [m]")
    ax.grid(True, which="both"); ax.legend()
    fig.suptitle("Étape C3 : agent x réutilisé sur x et y, erreurs tournées par ψ (modèle ARX identifié)")
    fig.tight_layout()
    fig.savefig(SORTIE / "sim_xy.png", dpi=120)
    print(f"-> {SORTIE}")


if __name__ == "__main__":
    main()
