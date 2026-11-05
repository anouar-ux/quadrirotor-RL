"""
COMPARAISONS DES TROIS FAMILLES DE CONTRÔLEURS, axe par axe puis sur la mission complète :
  1. LQI « double intégrateur » : modèle double intégrateur (gain k identifié), même loi que commande_LQ/gains_LQI.m, avec les
     gains RÉAJUSTÉS pour un contrôleur à 20 Hz (identification/reajuster_gains_LQI.py : réponse calée sur celle des
     modèles Simulink du LQI, où le LQI est continu ; les gains d'origine, réglés pour ce LQI continu,
     oscillent (z) ou divergent (theta) une fois échantillonnés à 20 Hz) ;
  2. LQ sur modèle ARX identifié (identification/lq_arx.py, commande_LQ/LQ_ARX_attitude/) : la dynamique des moteurs
     est dans le modèle ; pour x : PD réglé sur ce modèle (0,3 / 0,45, avec anticipation de vitesse) ;
  3. agents RL (PPO) entraînés sur les modèles identifiés (GRU de z et du lacet, ARX de l'attitude).

À quoi sert ce script : produire, sur le MÊME drone simulé (modèles identifiés : GRU de z, GRU du lacet, ARX de
l'attitude + translation), et à la même cadence de 20 Hz que le contrôleur Simulink, les figures et le tableau qui
comparent ces contrôleurs :
    z        : échelon 0 -> 0,5 m, mission de la Figure 1 (0 -> 0,5 -> -0,25 m), PRBS longue 0,3 <-> 0,7 m
    lacet    : scan 360° (un tour en 8 s), PRBS longue de vitesse ±0,6 rad/s
    attitude : échelon de 0,1 rad sur l'ARX (LQI double intégrateur, LQ ARX « C1 » lent, LQ ARX « rapide »)
    x        : échelon 2 m, échelon 5 m (saturation), voiture 5,5 m devant à 0,3 m/s, PRBS longue 0 <-> 1,5 m
               (cascade LQI : PD 0,23 / 0,28 + le LQI theta réajusté ; PD ARX et agent : LQ ARX rapide)
    mission  : voiture rectiligne à 0,3 m/s (scénario de la mission LQ) : mission LQ (Simulink, cascade LQI,
               commande_LQ/mission_LQ/) contre mission RL (agents, sim_mission.py)
Mesures bruitées comme g_noise = 0,1 pour les indicateurs ; courbes tracées sans bruit (lisibilité).

    python python/evaluation/comparaisons_LQI_ARX_RL.py     ->  resultats/comparaisons/ (6 figures + json + tableau .md)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.io import loadmat  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import AGENT_PSI, AGENT_X, AGENT_Z, COMMANDE_LQ, RESULTATS  # noqa: E402
from env_psi import MISSION_SCAN, YawEnv, consigne_lacet  # noqa: E402
from env_x import THETA_MAX, U_MAX_ATT, charger_boucle_interne  # noqa: E402
from env_z import MISSION_FIG1, TS, DroneEnv, mission  # noqa: E402
from essais_prbs import (BRUIT_PSI, BRUIT_X, BRUIT_Z, Agent, LQIController, LQIPsi, axe_psi_long, axe_z_long,  # noqa: E402
                         fabrique_axe_x, indicateurs, vol as vol_prbs)
from lq_arx import LQARXpsi, LQIARXz  # noqa: E402
from reajuster_gains_LQI import gains_20Hz  # noqa: E402
import env_x  # noqa: E402
from reference_pd_x import SIGMA_G01, pd_anticipation, pd_cascade_lqi, vol as vol_x  # noqa: E402
import sim_mission as M  # noqa: E402
import voiture as V  # noqa: E402

SORTIE = RESULTATS / "comparaisons"
COUL = {"LQI": "tab:red", "LQI/2": "tab:orange", "ARX": "tab:green", "ARX lent": "tab:olive", "RL": "tab:blue"}
NOM_LQI = "LQI double intégrateur (gains réajustés à 20 Hz)"
R = {}


def bp(d, cle="bande_passante_hz"):
    """Bande passante à -3 dB ; si le gain ne descend jamais sous -3 dB dans la plage mesurable, borne inférieure."""
    v = d.get(cle)
    return d.get("bande_superieure_a_hz") if v is None or not np.isfinite(v) else v


def fmt_bp(d, cle="bande_passante_hz"):
    v = d.get(cle)
    return f"{v:.2f} Hz" if v is not None and np.isfinite(v) else f"> {d.get('bande_superieure_a_hz') or float('nan'):.2f} Hz"


def indic_echelon(t, y, cible, t0=0.0):
    s = t >= t0
    y, t = y[s], t[s] - t0
    i10, i90 = np.argmax(y >= 0.1 * cible), np.argmax(y >= 0.9 * cible)
    j = np.where(np.abs(y - cible) > 0.02 * abs(cible))[0]
    return dict(montee_s=float((i90 - i10) * TS) if y.max() >= 0.9 * cible else None,
                depassement_pct=float(100 * max(y.max() - cible, 0) / abs(cible)),
                temps_reponse_2pct_s=float(t[j[-1]] + TS) if len(j) and j[-1] < len(y) - 1 else None,
                erreur_finale=float(abs(y[-20:].mean() - cible)))


# ================================================================ 1. ALTITUDE z
def ctrl_z():
    return {"LQI": (NOM_LQI, lambda env: LQIController(gains=gains_20Hz("z"))),
            "ARX": ("LQI sur modèle ARX", lambda env: LQIARXz()),
            "RL": ("agent PPO z", lambda env: Agent(AGENT_Z / "best_model"))}


def vol_z(fab, zc, bruit=False, n=200, graine=0):
    env = DroneEnv(z_ref_fixe=zc, **(BRUIT_Z if bruit else {}))
    obs, info = env.reset(seed=graine); c = fab(env); c.reset(info["m"])
    Z, U = [], []
    for _ in range(n):
        u = c.act(obs, info["m"]); obs, *_, info = env.step(np.array([u], np.float32)); Z.append(env.z); U.append(u)
    return np.arange(1, n + 1) * TS, np.array(Z), np.array(U)


def comparer_z():
    C = ctrl_z(); res = {}
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    t = np.arange(1, 201) * TS
    zr1 = np.array([mission(x, MISSION_FIG1) for x in t])
    for k, (nom, fab) in C.items():
        r = {}
        tt, Z, U = vol_z(fab, 0.5)
        r["echelon"] = indic_echelon(tt, Z, 0.5)
        r["echelon"]["agitation_u_bruit"] = float(np.mean(np.abs(np.diff(vol_z(fab, 0.5, True)[2]))))
        ax[0, 0].plot(tt, Z, color=COUL[k], label=nom); ax[1, 0].plot(tt, U, color=COUL[k], lw=0.9)
        _, Z1, _ = vol_z(fab, MISSION_FIG1); _, Z1b, _ = vol_z(fab, MISSION_FIG1, True)
        r["figure1_rmse_m"] = float(np.sqrt(np.mean((zr1 - Z1) ** 2)))
        r["figure1_rmse_bruit_m"] = float(np.sqrt(np.mean((zr1 - Z1b) ** 2)))
        ax[0, 1].plot(t, Z1, color=COUL[k], label=nom)
        Tp = vol_prbs(axe_z_long, fab, True)
        ind, _ = indicateurs(Tp)
        r["prbs_bruit"] = dict(rmse_m=ind["rmse"], bande_passante_hz=ind["bande_passante_hz"],
                               bande_superieure_a_hz=ind["bande_superieure_a_hz"], agitation=ind["agitation"])
        res[nom] = r
    ax[0, 0].axhline(0.5, color="k", ls=":"); ax[0, 0].set_title("Échelon 0 -> 0,5 m (sans bruit)"); ax[0, 0].set_ylabel("z [m]")
    ax[1, 0].set_ylabel("commande u"); ax[1, 0].set_xlabel("t [s]"); ax[1, 0].set_title("Commande pendant l'échelon")
    ax[0, 1].plot(t, zr1, "k:", label="consigne lissée"); ax[0, 1].set_title("Mission de la Figure 1 du sujet (sans bruit)")
    ax[0, 1].set_ylabel("z [m]"); ax[0, 1].set_xlabel("t [s]")
    barres(ax[1, 1], res, [("RMSE Fig. 1 bruit [cm]", lambda r: 100 * r["figure1_rmse_bruit_m"]),
                           ("RMSE PRBS bruit [cm]", lambda r: 100 * r["prbs_bruit"]["rmse_m"]),
                           ("bande passante [Hz] (borne inf. si >)", lambda r: bp(r["prbs_bruit"])),
                           ("dépassement échelon [%]", lambda r: r["echelon"]["depassement_pct"])], list(C))
    for a in ax.ravel()[:3]:
        a.grid(alpha=.3)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Altitude z : LQI double intégrateur / LQI sur modèle ARX / agent PPO (drone simulé : GRU identifié, 20 Hz)")
    fig.tight_layout(); fig.savefig(SORTIE / "comparaison_z.png", dpi=115); plt.close(fig)
    R["z"] = res


# ================================================================ 2. LACET psi
def ctrl_psi():
    return {"LQI": (NOM_LQI, lambda env: LQIPsi(gains=gains_20Hz("psi"))),
            "ARX": ("LQ sur modèle ARX", lambda env: LQARXpsi(w_ref_fn=lambda t, e=env: consigne_lacet(t, e.sauts)[1])),
            "RL": ("agent PPO lacet", lambda env: Agent(AGENT_PSI / "best_model"))}


def comparer_psi():
    C = ctrl_psi(); res = {}
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    for k, (nom, fab) in C.items():
        r = {}
        for bruit in (False, True):
            env = YawEnv(sauts_fixes=MISSION_SCAN, **(BRUIT_PSI if bruit else {}))
            obs, info = env.reset(seed=0); c = fab(env)
            try:
                c.reset(info["m"])
            except TypeError:
                c.reset()
            P, Rf, A = [], [], []
            for i in range(400):
                a = c.act(obs, info["m"]); obs, *_, info = env.step([a])
                P.append(env.psi); Rf.append(consigne_lacet(env.k * TS, env.sauts)[0]); A.append(a)
            P, Rf, t = np.array(P), np.array(Rf), np.arange(1, 401) * TS
            e = Rf - P; sc = (t > 6) & (t < 12)
            r["scan_bruit" if bruit else "scan"] = dict(rmse_rad=float(np.sqrt(np.mean(e ** 2))), erreur_max_rad=float(np.abs(e).max()),
                                                         erreur_pendant_scan_deg=float(np.degrees(np.sqrt(np.mean(e[sc] ** 2)))),
                                                         depassement_fin_deg=float(np.degrees(max(P[t > 13].max() - 2 * np.pi, 0))))
            if not bruit:
                ax[0, 0].plot(t, P, color=COUL[k], label=nom); ax[1, 0].plot(t, np.degrees(e), color=COUL[k], lw=1)
        Tp = vol_prbs(axe_psi_long, fab, True)
        ind, _ = indicateurs(Tp, "w_ref", "w"); ind_a, _ = indicateurs(Tp)
        r["prbs_bruit"] = dict(rmse_rad=ind_a["rmse"], bande_passante_vitesse_hz=ind["bande_passante_hz"],
                               bande_superieure_a_hz=ind["bande_superieure_a_hz"])
        res[nom] = r
    t = np.arange(1, 401) * TS
    ax[0, 0].plot(t, [consigne_lacet(x, MISSION_SCAN)[0] for x in t], "k:", label="consigne (un tour en 8 s)")
    ax[0, 0].set_ylabel("ψ [rad]"); ax[0, 0].set_title("Scan 360° (sans bruit)"); ax[0, 0].legend(fontsize=8)
    ax[1, 0].set_ylabel("erreur de cap [°]"); ax[1, 0].set_xlabel("t [s]"); ax[1, 0].set_title("Erreur de cap")
    barres(ax[0, 1], res, [("erreur pendant le scan [°]", lambda r: r["scan_bruit"]["erreur_pendant_scan_deg"]),
                           ("dépassement en fin de tour [°]", lambda r: r["scan"]["depassement_fin_deg"]),
                           ("RMSE PRBS bruit [°]", lambda r: np.degrees(r["prbs_bruit"]["rmse_rad"]))], list(C))
    barres(ax[1, 1], res, [("bande passante en vitesse [Hz] (borne inf. si >)", lambda r: bp(r["prbs_bruit"], "bande_passante_vitesse_hz"))], list(C))
    for a in (ax[0, 0], ax[1, 0]):
        a.grid(alpha=.3)
    fig.suptitle("Lacet ψ : LQI double intégrateur / LQ sur modèle ARX / agent PPO (drone simulé : GRU identifié, 20 Hz)")
    fig.tight_layout(); fig.savefig(SORTIE / "comparaison_psi.png", dpi=115); plt.close(fig)
    R["psi"] = res


# ================================================================ 3. ATTITUDE theta (sur l'ARX identifié)
def comparer_attitude():
    p_rap = charger_boucle_interne("rapide"); p_c1 = charger_boucle_interne("C1")
    a, b = p_rap["a"], p_rap["b"]
    kp, kd, ki = gains_20Hz("theta")

    def vol(loi, n=120, ref=0.1):
        th = w = wd = wd1 = I = 0.0; Uh = np.zeros(5); T, Uo = [], []
        for _ in range(n):
            u = float(np.clip(loi(th, w, wd, wd1, Uh, I, ref), -U_MAX_ATT, U_MAX_ATT))
            wdn = a[0] * wd + a[1] * wd1 + b[0] * u + b[1:] @ Uh
            wn = w + TS / 2 * (wd + wdn); thn = th + TS / 2 * (w + wn); I += TS * (ref - th)
            Uh = np.r_[u, Uh[:4]]; wd1, wd, w, th = wd, wdn, wn, thn; T.append(th); Uo.append(u)
        return np.arange(1, n + 1) * TS, np.array(T), np.array(Uo)
    lois = {"LQI": (NOM_LQI, lambda th, w, wd, wd1, U, I, r: kp * (r - th) - kd * w + ki * I),
            "ARX lent": ("LQ sur ARX, réglage C1 (lent)", lambda th, w, wd, wd1, U, I, r: -p_c1["K"] @ np.r_[th - r, w, wd, wd1, U]),
            "ARX": ("LQ sur ARX, réglage rapide (boucle interne)", lambda th, w, wd, wd1, U, I, r: -p_rap["K"] @ np.r_[th - r, w, wd, wd1, U])}
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8)); res = {}
    for k, (nom, loi) in lois.items():
        t, T, U = vol(loi)
        ind = indic_echelon(t, T, 0.1)
        ind["diverge"] = bool(np.abs(T).max() > 1.0)
        res[nom] = ind
        ax[0].plot(t, T, color=COUL[k], label=nom + (" — DIVERGE" if ind["diverge"] else ""))
        ax[1].step(t, U, color=COUL[k], lw=0.9, where="post")
    ax[0].axhline(0.1, color="k", ls=":"); ax[0].set_ylim(-0.01, 0.16); ax[0].set_ylabel("θ [rad]")
    ax[0].set_title("Échelon de 0,1 rad sur l'ARX identifié, commande à 20 Hz"); ax[0].legend(fontsize=8)
    ax[1].set_ylabel("commande différentielle u_θ"); ax[1].set_title("Commande différentielle")
    for x_ in ax:
        x_.grid(alpha=.3); x_.set_xlabel("t [s]")
    fig.suptitle("Attitude θ (et φ, même modèle) sur l'ARX identifié, 20 Hz : LQI double intégrateur (réajusté) / LQ sur modèle ARX")
    fig.tight_layout(); fig.savefig(SORTIE / "comparaison_attitude.png", dpi=115); plt.close(fig)
    R["attitude"] = res


# ================================================================ 4. POSITION x
def comparer_x():
    import stable_baselines3 as sb3
    ag = sb3.PPO.load(AGENT_X / "best_model", device="cpu")
    pol_ag = lambda obs, m, v: float(np.asarray(ag.predict(obs, deterministic=True)[0]).reshape(-1)[0])
    C = {"LQI": ("cascade LQI : PD 0,23/0,28 + LQI θ réajusté", pd_cascade_lqi(0.23, 0.28)),
         "ARX": ("PD réglé sur l'ARX : 0,3/0,45 + anticipation", pd_anticipation(0.3, 0.45)),
         "RL": ("agent PPO x (entraîné aux saturations)", pol_ag)}
    SC = {"saut_2m": ((1.0, "p", 2.0),), "saut_5m": ((1.0, "p", 5.0),), "voiture": ((1.0, "p", 5.5), (1.0, "v", 0.3))}
    fig, ax = plt.subplots(2, 2, figsize=(14, 8)); res = {}
    for k, (nom, pol) in C.items():
        r = {}
        env_x.BoucleTangage = BoucleLQICascade if k == "LQI" else BOUCLE_LQ   # boucle interne de la cascade
        for j, (sc, ev) in enumerate(SC.items()):
            T, _ = vol_x(pol, ev, sigma=0.0)
            Tb, _ = vol_x(pol, ev, sigma=SIGMA_G01)
            t, x, xr = T["t"], T["x"], T["x_ref"]
            e = xr - x
            if sc.startswith("saut"):
                cible = ev[0][2]
                r[sc] = indic_echelon(t, x, cible, 1.0)
            else:
                i = t > 12
                r[sc] = dict(depassement_voiture_m=float(max((x - xr)[t > 3].max(), 0)), retard_suivi_m=float(np.abs(e[i]).mean()))
            r[sc]["rmse_bruit_m"] = float(np.sqrt(np.mean((Tb["x_ref"] - Tb["x"]) ** 2)))
            ax.ravel()[j].plot(t, x, color=COUL[k], label=nom)
            if k == "RL":
                ax.ravel()[j].plot(t, xr, "k:", label="consigne")
        fab = lambda env, p=pol: _PolX(p, env)
        Tp = vol_prbs(fabrique_axe_x(7, 1.5, 1.5, 1), fab, True)
        ind, _ = indicateurs(Tp)
        r["prbs_bruit"] = dict(rmse_m=ind["rmse"], bande_passante_hz=ind["bande_passante_hz"],
                               bande_superieure_a_hz=ind["bande_superieure_a_hz"])
        res[nom] = r
    env_x.BoucleTangage = BOUCLE_LQ
    for a_, tit in zip(ax.ravel()[:3], ("Échelon de 2 m", "Échelon de 5 m (θ sature)", "Voiture 5,5 m devant, 0,3 m/s")):
        a_.set_title(tit); a_.grid(alpha=.3); a_.set_ylabel("x [m]"); a_.set_xlabel("t [s]")
    ax[0, 0].legend(fontsize=8)
    barres(ax[1, 1], res, [("dépass. 5 m [%]", lambda r: r["saut_5m"]["depassement_pct"]),
                           ("dépass. voiture [cm]", lambda r: 100 * r["voiture"]["depassement_voiture_m"]),
                           ("RMSE PRBS bruit [cm]", lambda r: 100 * r["prbs_bruit"]["rmse_m"]),
                           ("bande passante [Hz] x10", lambda r: 10 * bp(r["prbs_bruit"])),
                           ("retard sur la voiture [cm]", lambda r: 100 * r["voiture"]["retard_suivi_m"])], list(C))
    fig.suptitle("Position x : cascade LQI (PD + LQI θ réajusté) / PD réglé sur l'ARX + LQ ARX / agent PPO + LQ ARX")
    fig.tight_layout(); fig.savefig(SORTIE / "comparaison_x.png", dpi=115); plt.close(fig)
    R["x"] = res


BOUCLE_LQ = env_x.BoucleTangage


class BoucleLQICascade(env_x.BoucleTangage):
    """Boucle interne de la cascade LQI : son LQI theta (gains réajustés à 20 Hz) à la place du LQ ARX.
    Même physique que env_x.BoucleTangage (ARX de l'attitude + translation) ; seule la loi de commande change :
    u = Kp (theta_ref - theta) - Kd wy + Ki * intégrale(theta_ref - theta), bornée à +-U_MAX_ATT."""

    def reset(self):
        super().reset(); self.I = 0.0

    def step(self, theta_ref, biais=0.0, bruit=(0.0, 0.0, 0.0)):
        kp, kd, ki = gains_20Hz("theta")
        n_th, n_wy, _ = bruit
        e = theta_ref - (self.theta + biais + n_th); self.I += e * TS
        u = float(np.clip(kp * e - kd * (self.wy + n_wy) + ki * self.I, -U_MAX_ATT, U_MAX_ATT))
        p = self.p
        wdy_next = p["a"][0] * self.wdy + p["a"][1] * self.wdy_prev + p["b"][0] * u + p["b"][1:] @ self.U
        wy_next = self.wy + TS / 2 * (self.wdy + wdy_next)
        theta_next = self.theta + TS / 2 * (self.wy + wy_next)
        g, f = p["g_x"], p["frot_x"]
        vx_next = (self.vx + TS / 2 * (self.ax + g * theta_next)) / (1.0 - TS / 2 * f)
        self.ax = g * theta_next + f * vx_next
        self.x = self.x + TS / 2 * (self.vx + vx_next)
        self.U = np.r_[u, self.U[:4]]
        self.wdy_prev, self.wdy, self.wy, self.theta, self.vx = self.wdy, wdy_next, wy_next, theta_next, vx_next
        return u


class _PolX:
    def __init__(self, pol, env):
        self.pol, self.env = pol, env

    def reset(self, m=None):
        pass

    def act(self, obs, m):
        from env_x import consigne_x
        return self.pol(obs, m, consigne_x(m["t"], self.env.ev)[1])


# ================================================================ 5. MISSION (voiture rectiligne : scénario de la mission LQ)
def comparer_mission():
    S = loadmat(COMMANDE_LQ / "mission_LQ" / "resultats_mission_LQ.mat", squeeze_me=True, struct_as_record=False)["R"]
    lq = {}
    for r in np.atleast_1d(S):
        s = r.S; t = s.t; ph = s.phase; j = np.argmax(ph >= 5) if (ph >= 5).any() else None
        lq[f"g={r.g:g}"] = dict(t_pose_s=float(t[j]) if j is not None else None,
                                 ecart_contact_m=float(abs(s.x[j] - s.x_v[j])) if j is not None else None,
                                 vitesse_relative_ms=float(abs(s.vx[j] - 0.3)) if j is not None else None, L=s)
    tab = V.trajectoire(dict(t_depart=0.0, x0=1.0, y0=0.0, cap0=0.0, vitesse=((0.0, 0.05, 0.3),), courbure=()), t_fin=61.0)
    P = M.charger_politiques()
    rl = {}
    for sig, g in ((0.0, "g=0"), (M.SIGMA_G01, "g=0,1")):
        L = M.vol(P, tab, "rl", sig, graine=1, t_fin=50.0, **M.CONFIG_FINALE)
        d = M.bilan_complet(L, tab); d["L"] = L; rl[g] = d
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    for g, d in lq.items():
        s = d["L"]
        ax[0, 0].plot(s.t, s.x - s.x_v, color="tab:red", alpha=0.4 + 0.6 * (g == "g=0"), lw=1, label=f"mission LQ (Simulink), {g}")
        ax[0, 1].plot(s.t, s.z, color="tab:red", alpha=0.4 + 0.6 * (g == "g=0"), lw=1, label=f"mission LQ, {g}")
    for g, d in rl.items():
        L = d["L"]; t = L["v"][:, 0]
        ax[0, 0].plot(t, L["vrai"][:, 0] - L["y"][:, 9], color="tab:blue", alpha=0.4 + 0.6 * (g == "g=0"), lw=1, label=f"mission RL (agents), {g}")
        ax[0, 1].plot(t, L["vrai"][:, 2], color="tab:blue", alpha=0.4 + 0.6 * (g == "g=0"), lw=1, label=f"mission RL, {g}")
    ax[0, 0].set_ylim(-6, 1); ax[0, 0].set_ylabel("x drone − x voiture [m]"); ax[0, 0].set_title("Rattrapage de la voiture (0,3 m/s)")
    ax[0, 1].set_ylabel("z [m]"); ax[0, 1].set_title("Altitude : décollage, scan, poursuite, posé")
    for a_ in ax[0]:
        a_.grid(alpha=.3); a_.legend(fontsize=7); a_.set_xlabel("t [s]")
    noms = [f"LQ {g}" for g in lq] + [f"RL {g}" for g in rl]
    vals = [(d["t_pose_s"], d["ecart_contact_m"], d["vitesse_relative_ms"]) for d in lq.values()] + \
           [(d["t_posé"], d["ecart_contact_m"], d["vitesse_relative_horiz_ms"]) for d in rl.values()]
    cols = ["tab:red"] * len(lq) + ["tab:blue"] * len(rl)
    ax[1, 0].bar(noms, [v[0] or np.nan for v in vals], color=cols); ax[1, 0].set_ylabel("instant du posé [s]")
    ax[1, 1].bar(np.arange(len(noms)) - 0.2, [100 * (v[1] or np.nan) for v in vals], 0.4, color=cols, label="écart au contact [cm]")
    ax[1, 1].bar(np.arange(len(noms)) + 0.2, [100 * (v[2] or np.nan) for v in vals], 0.4, color=cols, alpha=0.5, hatch="//",
                 label="vitesse relative [cm/s]")
    ax[1, 1].set_xticks(range(len(noms))); ax[1, 1].set_xticklabels(noms); ax[1, 1].legend(fontsize=8)
    for a_ in ax[1]:
        a_.grid(alpha=.3, axis="y")
    fig.suptitle("Mission d'atterrissage, même voiture rectiligne (x_v0 = 1 m, 0,3 m/s) : cascade LQI double intégrateur / agents RL")
    fig.tight_layout(); fig.savefig(SORTIE / "comparaison_mission.png", dpi=115); plt.close(fig)
    R["mission"] = {"LQ (Simulink, mission_LQ.slx)": {g: {k: v for k, v in d.items() if k != "L"} for g, d in lq.items()},
                    "RL (agents, simulateur assemblé)": {g: {k: v for k, v in d.items() if k in ("t_posé", "ecart_contact_m",
                                                         "vitesse_relative_horiz_ms", "derive_scan_m", "erreur_z_vol_m")}
                                                         for g, d in rl.items()}}


# ================================================================ outils
def barres(ax, res, metriques, cles):
    noms = list(res); n = len(noms); x = np.arange(len(metriques)); w = 0.8 / n
    for i, nom in enumerate(noms):
        v = [f(res[nom]) for _, f in metriques]
        ax.bar(x + (i - (n - 1) / 2) * w, v, w, color=COUL[cles[i]], label=nom)
    ax.set_xticks(x); ax.set_xticklabels([m for m, _ in metriques], fontsize=7, rotation=12, ha="right"); ax.grid(alpha=.3, axis="y")
    ax.legend(fontsize=7); ax.set_title("Indicateurs (bruit de mesure g_noise = 0,1)")


def figure_synthese():
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(wspace=0.55))
    z = R["z"]; p = R["psi"]; x = R["x"]
    for a_, (tit, d, f, unite) in zip(ax, (
            ("z : RMSE sur la PRBS longue", z, lambda r: 100 * r["prbs_bruit"]["rmse_m"], "cm"),
            ("lacet : erreur pendant le scan", p, lambda r: r["scan_bruit"]["erreur_pendant_scan_deg"], "°"),
            ("x : RMSE sur la PRBS longue", x, lambda r: 100 * r["prbs_bruit"]["rmse_m"], "cm"))):
        noms = list(d); v = [f(d[n]) for n in noms]
        court = [n.split(" :")[0].replace(" (entraîné aux saturations)", "") for n in noms]
        coul = ["tab:red" if ("double intégrateur" in n or "cascade LQI" in n) else "tab:green" if "ARX" in n else "tab:blue" for n in noms]
        a_.barh(court, v, color=coul); a_.set_xlabel(unite); a_.set_title(tit); a_.grid(alpha=.3, axis="x")
        for i, val in enumerate(v):
            a_.text(val, i, f" {val:.2f}", va="center", fontsize=8)
    fig.suptitle("Synthèse (mesures bruitées g_noise = 0,1) — rouge : LQI double intégrateur (réajusté à 20 Hz), vert : LQ sur modèle ARX, bleu : agent RL")
    fig.tight_layout(); fig.savefig(SORTIE / "comparaison_synthese.png", dpi=115); plt.close(fig)


def tableau_md():
    L = ["# Comparaisons LQI double intégrateur / LQ sur modèle ARX / agents RL", "",
         "Drone simulé : modèles identifiés (GRU de z et du lacet, ARX de l'attitude), contrôleurs à 20 Hz, bruit g_noise = 0,1 "
         "pour les indicateurs « bruit ». Produit par python/evaluation/comparaisons_LQI_ARX_RL.py.", "",
         "LQI double intégrateur : même loi que gains_LQI.m, gains réajustés pour 20 Hz (réponse calée sur Simulink, "
         "voir reajustement_gains_LQI.png et python/modeles/gains_LQI_20Hz.json).", ""]
    L += ["## Altitude z", "", "| contrôleur | montée | dépassement | RMSE Fig. 1 (bruit) | RMSE PRBS (bruit) | bande passante |", "|---|---|---|---|---|---|"]
    for n, r in R["z"].items():
        e = r["echelon"]
        L.append(f"| {n} | {e['montee_s'] or float('nan'):.2f} s | {e['depassement_pct']:.1f} % | {100 * r['figure1_rmse_bruit_m']:.1f} cm | "
                 f"{100 * r['prbs_bruit']['rmse_m']:.1f} cm | {fmt_bp(r['prbs_bruit'])} |")
    L += ["", "## Lacet", "", "| contrôleur | erreur pendant le scan (bruit) | dépassement en fin de tour | RMSE PRBS (bruit) | bande passante (vitesse) |", "|---|---|---|---|---|"]
    for n, r in R["psi"].items():
        L.append(f"| {n} | {r['scan_bruit']['erreur_pendant_scan_deg']:.2f}° | {r['scan']['depassement_fin_deg']:.1f}° | "
                 f"{np.degrees(r['prbs_bruit']['rmse_rad']):.1f}° | {fmt_bp(r['prbs_bruit'], 'bande_passante_vitesse_hz')} |")
    L += ["", "## Attitude θ (échelon 0,1 rad sur l'ARX, 20 Hz)", "", "| contrôleur | montée | dépassement | temps de réponse 2 % | diverge |", "|---|---|---|---|---|"]
    for n, r in R["attitude"].items():
        f = lambda v, u: (f"{v:.2f} {u}" if v is not None else "—")
        L.append(f"| {n} | {f(r['montee_s'], 's')} | {r['depassement_pct']:.0f} % | {f(r['temps_reponse_2pct_s'], 's')} | {'OUI' if r['diverge'] else 'non'} |")
    L += ["", "## Position x (cascade LQI : PD + le LQI θ réajusté ; PD ARX et agent : LQ ARX rapide en boucle interne)", "", "| contrôleur | dépassement 2 m | dépassement 5 m | dépassement voiture | retard sur la voiture | RMSE PRBS (bruit) | bande passante |", "|---|---|---|---|---|---|---|"]
    for n, r in R["x"].items():
        L.append(f"| {n} | {r['saut_2m']['depassement_pct']:.1f} % | {r['saut_5m']['depassement_pct']:.1f} % | "
                 f"{100 * r['voiture']['depassement_voiture_m']:.1f} cm | {100 * r['voiture']['retard_suivi_m']:.1f} cm | "
                 f"{100 * r['prbs_bruit']['rmse_m']:.1f} cm | {fmt_bp(r['prbs_bruit'])} |")
    L += ["", "## Mission (voiture rectiligne 0,3 m/s)", "", "| mission | bruit | posé à | écart au contact | vitesse relative |", "|---|---|---|---|---|"]
    for fam, d in R["mission"].items():
        for g, v in d.items():
            tp = v.get("t_pose_s", v.get("t_posé")); vr = v.get("vitesse_relative_ms", v.get("vitesse_relative_horiz_ms"))
            L.append(f"| {fam} | {g} | {tp:.1f} s | {100 * v['ecart_contact_m']:.1f} cm | {100 * vr:.1f} cm/s |")
    (SORTIE / "comparaisons.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)
    for nom, f in (("z", comparer_z), ("lacet", comparer_psi), ("attitude", comparer_attitude), ("x", comparer_x),
                   ("mission", comparer_mission)):
        print(f"... {nom}", flush=True); f()
    figure_synthese(); tableau_md()
    with open(SORTIE / "comparaisons.json", "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False, default=float)
    print((SORTIE / "comparaisons.md").read_text(encoding="utf-8"))
    print(f"-> {SORTIE}")


if __name__ == "__main__":
    main()
