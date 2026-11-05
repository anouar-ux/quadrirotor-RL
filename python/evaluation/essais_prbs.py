"""
ESSAIS PRBS des contrôleurs : agents RL comparés aux contrôleurs classiques (LQI double intégrateur, cascade LQ + PD).

À quoi sert ce script : les agents ont été évalués sur des échelons ; une PRBS (séquence binaire pseudo-aléatoire de
longueur maximale) excite toute une bande de fréquences. Elle vérifie que l'agent suit aussi des consignes qui
changent souvent, et permet d'estimer la bande passante en boucle fermée (|sortie / consigne| en fréquence).

Deux séries d'essais (fusion des anciens essais_prbs.py et essais_prbs_longs.py) :

    python python/evaluation/essais_prbs.py            # essais LONGS (2 à 4 min par essai)  -> resultats/essais_prbs_long/
    python python/evaluation/essais_prbs.py --court    # essais COURTS (~1 min par essai)    -> resultats/essais_prbs_court/

PRBS adaptée à chaque dynamique : la période de bit Tb est de l'ordre du temps de montée en boucle fermée, pour que
l'on voie à la fois des paliers courts (hautes fréquences) et longs (le drone a le temps d'arriver). La consigne PRBS
passe par le même lissage que pendant l'entraînement. Chaque essai est fait sans bruit et avec un bruit de mesure
équivalent à g_noise = 0,1.

Essais COURTS (2 périodes d'une PRBS 5-6 bits, ~1 min) :
    z     : hauteur 0,3 <-> 0,7 m autour de 0,5 m,       Tb = 0,5 s, PRBS 6 bits (63 bits = 31,5 s), 2 périodes
    lacet : vitesse de lacet -0,6 <-> +0,6 rad/s,        Tb = 0,5 s, PRBS 6 bits, 2 périodes (variable entraînée)
    x     : position 0 <-> 1,5 m,                        Tb = 1,5 s, PRBS 5 bits (31 bits = 46,5 s), 2 périodes

Essais LONGS (PRBS 7 bits, 2 à 4 min de vol) : estimation plus fiable de la réponse en fréquence (plus de segments
moyennés) et vérification que les agents restent stables et précis sur un vol long (pas de dérive de l'intégrale) :
    z        : hauteur 0,3 <-> 0,7 m,     Tb = 0,5 s, PRBS 7 bits, 2 périodes (127 s)
                agent PPO z  /  LQI (gains / 2)
    lacet    : vitesse ±0,6 rad/s,        Tb = 0,5 s, PRBS 7 bits, 2 périodes (127 s)
                agent PPO lacet  /  LQI de lacet
    x fin    : position 0 <-> 1,5 m,      Tb = 1,5 s, PRBS 7 bits, 1 période (190,5 s)   (consignes faisables)
    x grand  : position 0 <-> 6 m,        Tb = 4 s,   PRBS 6 bits, 1 période (252 s)     (theta_ref sature)
                agent x saturations / agent x précédent / cascade LQ + PD optimisé (0,3 / 0,45, avec anticipation)
                / cascade LQI : LQ + PD 0,23 / 0,28, sans anticipation

"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import RESULTATS  # noqa: E402  (rend importables les sous-dossiers de python/)
from env_z import TS, DroneEnv, mission  # noqa: E402
from reference_pd_x import SIGMA_G01, pd_anticipation, pd_cascade_lqi  # noqa: E402
from train_ppo_psi import BRUIT_ENTRAINEMENT as BRUIT_PSI, LQIPsi  # noqa: E402
from train_sac_z import BRUIT_ENTRAINEMENT as BRUIT_Z, LQIController  # noqa: E402
from env_x import THETA_MAX, XEnv, consigne_x  # noqa: E402
from env_psi import U_PSI_MAX, YawEnv, consigne_lacet  # noqa: E402

SORTIE_COURT = RESULTATS / "essais_prbs_court"
SORTIE_LONG = RESULTATS / "essais_prbs_long"
AGENT_Z = RESULTATS / "agent_z_ppo" / "best_model"
AGENT_PSI = RESULTATS / "agent_psi_ppo" / "best_model"
AGENT_X_SAT = RESULTATS / "agent_x_ppo_saturation" / "best_model"
AGENT_X_ANCIEN = RESULTATS / "agent_x_ppo" / "best_model"


def rel(p):
    """Chemin relatif au dossier du projet (pas de chemin personnel dans les .json publiés)."""
    p = Path(p).resolve()
    return p.relative_to(RESULTATS.parents[1]).as_posix() if p.is_relative_to(RESULTATS.parents[1]) else p.name
BRUIT_X = dict(sigma_x=SIGMA_G01, sigma_vx=SIGMA_G01, sigma_theta=SIGMA_G01, sigma_wy=SIGMA_G01)
PD_X = (0.3, 0.45)


def prbs(n_bits):
    """Séquence binaire de longueur maximale 2^n - 1 (registre à décalage, polynômes primitifs classiques)."""
    taps = {5: (5, 3), 6: (6, 5), 7: (7, 6)}[n_bits]
    etat, seq = [1] * n_bits, []
    for _ in range(2 ** n_bits - 1):
        seq.append(etat[-1])
        nouveau = etat[taps[0] - 1] ^ etat[taps[1] - 1]
        etat = [nouveau] + etat[:-1]
    return np.array(seq)


def instants_niveaux(n_bits, tb, t0, bas, haut, n_periodes=2):
    """Liste (t, niveau) des CHANGEMENTS de niveau de la PRBS, à partir de t0."""
    bits = np.tile(prbs(n_bits), n_periodes)
    niveaux = np.where(bits == 1, haut, bas)
    ev, prec = [], None
    for i, v in enumerate(niveaux):
        if v != prec:
            ev.append((t0 + i * tb, float(v)))
            prec = v
    return ev, t0 + len(bits) * tb


# ------------------------------------------------------------------ adaptateurs : act(obs, m) -> action
class Agent:
    def __init__(self, chemin):
        self.model = PPO.load(chemin, device="cpu")

    def reset(self, m=None):
        pass

    def act(self, obs, m):
        return float(np.asarray(self.model.predict(obs, deterministic=True)[0]).reshape(-1)[0])


class PDx:
    def __init__(self, env):
        self.pol, self.env = pd_anticipation(*PD_X), env

    def reset(self, m=None):
        pass

    def act(self, obs, m):
        return self.pol(obs, m, consigne_x(m["t"], self.env.ev)[1])


# ------------------------------------------------------------------ les trois axes
def axe_z_court(bruit):
    ev, t_fin = instants_niveaux(6, 0.5, 3.0, 0.3, 0.7)
    paliers = [(0.0, 0.0), (0.0, 0.5)] + ev                # décollage à 0,5 m, puis PRBS à partir de 3 s
    fab = lambda: DroneEnv(z_ref_fixe=paliers, **(BRUIT_Z if bruit else {}))
    lire = lambda env: (env.z, mission(env.k * TS, env.z_ref))
    return fab, lire, t_fin, 3.0


def axe_psi_court(bruit):
    ev, t_fin = instants_niveaux(6, 0.5, 2.0, -0.6, 0.6)
    fab = lambda: YawEnv(sauts_fixes=ev, **(BRUIT_PSI if bruit else {}))
    lire = lambda env: (env.psi, consigne_lacet(env.k * TS, env.sauts)[0], env.wz, consigne_lacet(env.k * TS, env.sauts)[1])
    return fab, lire, t_fin, 2.0


def axe_x_court(bruit):
    ev_niv, t_fin = instants_niveaux(5, 1.5, 2.0, 0.0, 1.5)
    ev, prec = [], 0.0
    for t, v in ev_niv:
        if v != prec:
            ev.append((t, "p", v - prec)); prec = v
    fab = lambda: XEnv(evenements_fixes=ev, **(BRUIT_X if bruit else {}))
    lire = lambda env: (env.boucle.x, consigne_x(env.k * TS, env.ev)[0])
    return fab, lire, t_fin, 2.0


def vol(axe, fab_ctrl, bruit, graine=0):
    fab, lire, t_fin, t0 = axe(bruit)
    env = fab()
    obs, info = env.reset(seed=graine)
    ctrl = fab_ctrl(env)
    try:
        ctrl.reset(info["m"])
    except TypeError:                                 # LQIPsi.reset() ne prend pas d'argument
        ctrl.reset()
    T = {"t": [], "y": [], "ref": [], "a": [], "w": [], "w_ref": []}
    for _ in range(int(round(t_fin / TS))):
        a = ctrl.act(obs, info["m"])
        obs, _, _, _, info = env.step([a])          # au-delà de N_STEPS : on continue (la troncature est ignorée)
        v = lire(env)
        T["t"].append(env.k * TS); T["y"].append(v[0]); T["ref"].append(v[1]); T["a"].append(a)
        if len(v) > 2:
            T["w"].append(v[2]); T["w_ref"].append(v[3])
    T = {k: np.asarray(v) for k, v in T.items()}
    T["t0"] = t0
    return T


def indicateurs(T, entree="ref", sortie="y"):
    """Erreur de suivi, retard (intercorrélation) et bande passante à -3 dB (estimateur H1 = Pry / Prr)."""
    s = T["t"] >= T["t0"]
    r, y = T[entree][s], T[sortie][s]
    e = r - y
    rc, yc = r - r.mean(), y - y.mean()
    lags = np.arange(-40, 41)
    xc = [np.dot(rc[max(0, -l):len(rc) - max(0, l)], yc[max(0, l):len(yc) - max(0, -l)]) for l in lags]
    retard = float(lags[int(np.argmax(xc))] * TS)
    f, Prr = signal.welch(r, fs=1 / TS, nperseg=256)
    _, Pry = signal.csd(r, y, fs=1 / TS, nperseg=256)
    _, Coh = signal.coherence(r, y, fs=1 / TS, nperseg=256)
    H = Pry / np.maximum(Prr, 1e-15)
    gain = np.abs(H)
    ok = (Coh > 0.8) & (f > 0)
    sous = np.where(ok & (gain < 1 / np.sqrt(2)))[0]
    bp = float(f[sous[0]]) if len(sous) else float("nan")
    # si le gain reste au-dessus de -3 dB partout où la mesure est fiable (cohérence > 0,8) : bande > f_fiable
    f_fiable = float(f[ok].max()) if ok.any() else float("nan")
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), e_max=float(np.max(np.abs(e))), retard_s=retard,
                bande_passante_hz=bp, bande_superieure_a_hz=f_fiable if np.isnan(bp) else None,
                agitation=float(np.mean(np.abs(np.diff(T["a"]))))), (f, gain, Coh, np.angle(H))


AXES_COURT = {
    "z": dict(axe=axe_z_court, unite="m", nom_y="z", ctrl={"agent PPO": lambda env: Agent(AGENT_Z),
                                                    "LQI (gains / 2)": lambda env: LQIController()},
              cmd=("u", 1.0)),
    "psi": dict(axe=axe_psi_court, unite="rad", nom_y="ψ", ctrl={"agent PPO": lambda env: Agent(AGENT_PSI),
                                                          "LQI lacet": lambda env: LQIPsi()},
                cmd=("u_ψ", U_PSI_MAX)),
    "x": dict(axe=axe_x_court, unite="m", nom_y="x", ctrl={"agent PPO": None, f"PD {PD_X[0]}/{PD_X[1]}": lambda env: PDx(env)},
              cmd=("θ_ref [rad]", THETA_MAX)),
}


def main_court(dx=None):
    """Essais courts (z, lacet, x) : un agent x (par défaut l'agent x retenu) contre le PD."""
    dx = Path(dx) if dx else AGENT_X_SAT.parent
    AXES_COURT["x"]["ctrl"]["agent PPO"] = lambda env: Agent(dx / "best_model")
    SORTIE_COURT.mkdir(parents=True, exist_ok=True)
    resultats, npz = {"resultats": {"z": rel(AGENT_Z), "psi": rel(AGENT_PSI), "x": rel(dx)}}, {}
    figB, axB = plt.subplots(2, 3, figsize=(15, 6.5))
    for j, (cle, A) in enumerate(AXES_COURT.items()):
        resultats[cle] = {}
        fig, ax = plt.subplots(4 if cle == "psi" else 3, 1, figsize=(13, 10 if cle == "psi" else 8), sharex=True)
        for (nom, fab_ctrl), st in zip(A["ctrl"].items(), ("-", "--")):
            for bruit in (False, True):
                T = vol(A["axe"], fab_ctrl, bruit)
                ind, (f, gain, coh, phase) = indicateurs(T)
                if cle == "psi":   # la PRBS porte sur la vitesse : bande passante et Bode sur omega_z / omega_ref
                    ind_w, (f, gain, coh, phase) = indicateurs(T, "w_ref", "w")
                    ind.update({f"vitesse_{k}": v for k, v in ind_w.items()})
                    ind["bande_passante_hz"] = ind_w["bande_passante_hz"]
                    ind["bande_superieure_a_hz"] = ind_w["bande_superieure_a_hz"]
                lab = f"{nom}{' | bruit g~0,1' if bruit else ''}"
                resultats[cle][lab] = ind
                for k in ("t", "y", "ref", "a", "w", "w_ref"):
                    if len(T[k]):
                        npz[f"{cle}|{lab}|{k}"] = T[k]
                if not bruit:
                    ax[0].plot(T["t"], T["y"], st, label=nom); ax[1].plot(T["t"], T["ref"] - T["y"], st, label=nom)
                    ax[-1].plot(T["t"], A["cmd"][1] * T["a"], st, lw=0.9, label=nom)
                    if cle == "psi":
                        ax[2].plot(T["t"], T["w"], st, label=nom)
                    axB[0, j].semilogx(f[1:], 20 * np.log10(np.maximum(gain[1:], 1e-6)), st, label=nom)
                    axB[1, j].semilogx(f[1:], np.degrees(np.unwrap(phase[1:])), st, label=nom)
        ax[0].plot(T["t"], T["ref"], "k:", lw=1, label="consigne (PRBS lissée)")
        if cle == "psi":
            ax[2].plot(T["t"], T["w_ref"], "k:", lw=1, label="consigne de vitesse")
            ax[2].set_ylabel("ω_z [rad/s]")
        ax[0].set_ylabel(f"{A['nom_y']} [{A['unite']}]"); ax[1].set_ylabel(f"erreur [{A['unite']}]")
        ax[-1].set_ylabel(A["cmd"][0]); ax[-1].set_xlabel("t [s]")
        for x_ in ax:
            x_.grid(alpha=.3); x_.legend(fontsize=8, loc="upper right")
        ax[0].set_title({"z": "Altitude : PRBS 0,3 <-> 0,7 m (Tb = 0,5 s)",
                         "psi": "Lacet : PRBS de vitesse ±0,6 rad/s (Tb = 0,5 s)",
                         "x": "Position x : PRBS 0 <-> 1,5 m (Tb = 1,5 s)"}[cle] + ", sans bruit")
        fig.tight_layout(); fig.savefig(SORTIE_COURT / f"prbs_{cle}.png", dpi=110); plt.close(fig)
        axB[0, j].set_title({"z": "z / z_ref", "psi": "ω_z / ω_ref (lacet)", "x": "x / x_ref"}[cle])
        axB[0, j].axhline(-3, color="k", ls=":", lw=1); axB[0, j].set_ylabel("gain [dB]"); axB[0, j].set_ylim(-20, 6)
        axB[1, j].set_ylabel("phase [°]"); axB[1, j].set_xlabel("fréquence [Hz]")
        for x_ in axB[:, j]:
            x_.grid(alpha=.3, which="both"); x_.legend(fontsize=8)
    figB.suptitle("Réponse en fréquence en boucle fermée estimée à partir des essais PRBS (sans bruit)")
    figB.tight_layout(); figB.savefig(SORTIE_COURT / "prbs_bode.png", dpi=110)
    json.dump(resultats, open(SORTIE_COURT / "prbs_resultats.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    np.savez_compressed(SORTIE_COURT / "prbs_trajectoires.npz", **npz)
    # ---- tableau
    print(f"{'axe':<5}{'contrôleur':<32}{'RMSE':>9}{'écart max':>11}{'retard':>9}{'bande -3 dB':>13}{'agitation':>11}")
    for cle in AXES_COURT:
        for lab, r in resultats[cle].items():
            bp = (f"{r['bande_passante_hz']:>9.2f} Hz" if r["bande_superieure_a_hz"] is None
                  else f"  > {r['bande_superieure_a_hz']:.2f} Hz")
            print(f"{cle:<5}{lab:<32}{r['rmse']:>9.4f}{r['e_max']:>11.3f}{r['retard_s']:>8.2f}s{bp:>13}{r['agitation']:>11.4f}")
    print(f"\nÉcrit dans {SORTIE_COURT}")


# ====================================================================================================
# ESSAIS LONGS
# ====================================================================================================
class PolX:
    """Adapte une politique reference_pd_x (obs, m, v_ref) -> a à l'interface act(obs, m)."""

    def __init__(self, pol, env):
        self.pol, self.env = pol, env

    def reset(self, m=None):
        pass

    def act(self, obs, m):
        return self.pol(obs, m, consigne_x(m["t"], self.env.ev)[1])


def axe_z_long(bruit):
    ev, t_fin = instants_niveaux(7, 0.5, 3.0, 0.3, 0.7, n_periodes=2)
    paliers = [(0.0, 0.0), (0.0, 0.5)] + ev
    return (lambda: DroneEnv(z_ref_fixe=paliers, **(BRUIT_Z if bruit else {})),
            lambda env: (env.z, mission(env.k * TS, env.z_ref)), t_fin, 3.0)


def axe_psi_long(bruit):
    ev, t_fin = instants_niveaux(7, 0.5, 2.0, -0.6, 0.6, n_periodes=2)
    return (lambda: YawEnv(sauts_fixes=ev, **(BRUIT_PSI if bruit else {})),
            lambda env: (env.psi, consigne_lacet(env.k * TS, env.sauts)[0], env.wz, consigne_lacet(env.k * TS, env.sauts)[1]),
            t_fin, 2.0)


def fabrique_axe_x(n_bits, tb, haut, n_periodes):
    def axe(bruit):
        ev_niv, t_fin = instants_niveaux(n_bits, tb, 2.0, 0.0, haut, n_periodes=n_periodes)
        ev, prec = [], 0.0
        for t, v in ev_niv:
            if v != prec:
                ev.append((t, "p", v - prec)); prec = v
        return (lambda: XEnv(evenements_fixes=ev, **(BRUIT_X if bruit else {})),
                lambda env: (env.boucle.x, consigne_x(env.k * TS, env.ev)[0]), t_fin, 2.0)
    return axe


CTRL_X = {"agent x saturations": lambda env: Agent(AGENT_X_SAT),
          "agent x précédent": lambda env: Agent(AGENT_X_ANCIEN),
          "LQ + PD optimisé 0,3/0,45": lambda env: PolX(pd_anticipation(0.3, 0.45), env),
          "LQ + PD (cascade LQI) 0,23/0,28": lambda env: PolX(pd_cascade_lqi(0.23, 0.28), env)}

ESSAIS_LONG = {
    "z": dict(axe=axe_z_long, unite="m", nom_y="z", cmd=("u", 1.0), bode="z / z_ref",
              titre="Altitude : PRBS 7 bits 0,3 <-> 0,7 m, Tb = 0,5 s",
              ctrl={"agent PPO z": lambda env: Agent(AGENT_Z), "LQI z (gains / 2)": lambda env: LQIController()}),
    "psi": dict(axe=axe_psi_long, unite="rad", nom_y="ψ", cmd=("u_ψ", U_PSI_MAX), bode="ω_z / ω_ref",
                titre="Lacet : PRBS 7 bits de vitesse ±0,6 rad/s, Tb = 0,5 s",
                ctrl={"agent PPO lacet": lambda env: Agent(AGENT_PSI), "LQI lacet": lambda env: LQIPsi()}),
    "x_fin": dict(axe=fabrique_axe_x(7, 1.5, 1.5, 1), unite="m", nom_y="x", cmd=("θ_ref [rad]", THETA_MAX),
                  bode="x / x_ref (0-1,5 m)", titre="Position x : PRBS 7 bits 0 <-> 1,5 m, Tb = 1,5 s", ctrl=CTRL_X),
    "x_grand": dict(axe=fabrique_axe_x(6, 4.0, 6.0, 1), unite="m", nom_y="x", cmd=("θ_ref [rad]", THETA_MAX),
                    bode="x / x_ref (0-6 m, saturé)", titre="Position x : PRBS 6 bits 0 <-> 6 m, Tb = 4 s (θ sature)",
                    ctrl=CTRL_X),
}
STYLES = ("-", "--", "-.", ":")


def main_long():
    """Essais longs (z, lacet, x fin, x grand) : tous les contrôleurs RL et classiques."""
    SORTIE_LONG.mkdir(parents=True, exist_ok=True)
    resultats = {"agents": {"z": rel(AGENT_Z), "psi": rel(AGENT_PSI), "x_sat": rel(AGENT_X_SAT),
                            "x_ancien": rel(AGENT_X_ANCIEN)}}
    npz = {}
    figB, axB = plt.subplots(2, len(ESSAIS_LONG), figsize=(5 * len(ESSAIS_LONG), 6.5))
    for j, (cle, E) in enumerate(ESSAIS_LONG.items()):
        resultats[cle] = {}
        n_l = 4 if cle == "psi" else 3
        fig, ax = plt.subplots(n_l, 2, figsize=(17, 2.6 * n_l + 1), sharex="col",
                               gridspec_kw=dict(width_ratios=(2.2, 1)))
        for (nom, fab_ctrl), st in zip(E["ctrl"].items(), STYLES):
            for bruit in (False, True):
                T = vol(E["axe"], fab_ctrl, bruit)
                ind, (f, gain, coh, phase) = indicateurs(T)
                if cle == "psi":
                    ind_w, (f, gain, coh, phase) = indicateurs(T, "w_ref", "w")
                    ind.update({f"vitesse_{k}": v for k, v in ind_w.items()})
                    ind["bande_passante_hz"] = ind_w["bande_passante_hz"]
                    ind["bande_superieure_a_hz"] = ind_w["bande_superieure_a_hz"]
                # stabilité sur le vol long : erreur RMS sur le premier et le dernier quart
                s = T["t"] >= T["t0"]; e = (T["ref"] - T["y"])[s]; q = len(e) // 4
                ind["rmse_1er_quart"] = float(np.sqrt(np.mean(e[:q] ** 2)))
                ind["rmse_dernier_quart"] = float(np.sqrt(np.mean(e[-q:] ** 2)))
                ind["duree_s"] = float(T["t"][-1])
                lab = f"{nom}{' | bruit g~0,1' if bruit else ''}"
                resultats[cle][lab] = ind
                for k in ("t", "y", "ref", "a", "w", "w_ref"):
                    if len(T[k]):
                        npz[f"{cle}|{lab}|{k}"] = T[k]
                if bruit:
                    continue
                for c, zoom in ((0, None), (1, (T["t0"], T["t0"] + 40.0))):
                    m = np.ones_like(T["t"], bool) if zoom is None else (T["t"] >= zoom[0]) & (T["t"] <= zoom[1])
                    lw = 0.7 if c == 0 else 1.1
                    ax[0, c].plot(T["t"][m], T["y"][m], st, lw=lw, label=nom)
                    ax[1, c].plot(T["t"][m], (T["ref"] - T["y"])[m], st, lw=lw, label=nom)
                    ax[-1, c].plot(T["t"][m], E["cmd"][1] * T["a"][m], st, lw=0.6 if c == 0 else 0.9, label=nom)
                    if cle == "psi":
                        ax[2, c].plot(T["t"][m], T["w"][m], st, lw=lw, label=nom)
                axB[0, j].semilogx(f[1:], 20 * np.log10(np.maximum(gain[1:], 1e-6)), st, label=nom)
                axB[1, j].semilogx(f[1:], np.degrees(np.unwrap(phase[1:])), st, label=nom)
        for c, zoom in ((0, None), (1, (T["t0"], T["t0"] + 40.0))):
            m = np.ones_like(T["t"], bool) if zoom is None else (T["t"] >= zoom[0]) & (T["t"] <= zoom[1])
            ax[0, c].plot(T["t"][m], T["ref"][m], "k", lw=0.6, alpha=0.6, label="consigne (PRBS lissée)")
            if cle == "psi":
                ax[2, c].plot(T["t"][m], T["w_ref"][m], "k", lw=0.6, alpha=0.6, label="consigne de vitesse")
                ax[2, c].set_ylabel("ω_z [rad/s]")
            ax[0, c].set_ylabel(f"{E['nom_y']} [{E['unite']}]"); ax[1, c].set_ylabel(f"erreur [{E['unite']}]")
            ax[-1, c].set_ylabel(E["cmd"][0]); ax[-1, c].set_xlabel("t [s]")
            for a_ in ax[:, c]:
                a_.grid(alpha=.3)
        ax[0, 0].set_title(E["titre"] + " — vol complet, sans bruit")
        ax[0, 1].set_title("zoom sur les 40 premières secondes")
        ax[0, 1].legend(fontsize=7, loc="best")
        fig.tight_layout(); fig.savefig(SORTIE_LONG / f"prbs_long_{cle}.png", dpi=110); plt.close(fig)
        axB[0, j].set_title(E["bode"]); axB[0, j].axhline(-3, color="k", ls=":", lw=1)
        axB[0, j].set_ylabel("gain [dB]"); axB[0, j].set_ylim(-20, 8)
        axB[1, j].set_ylabel("phase [°]"); axB[1, j].set_xlabel("fréquence [Hz]")
        for a_ in axB[:, j]:
            a_.grid(alpha=.3, which="both")
        axB[0, j].legend(fontsize=7)
    figB.suptitle("Réponse en fréquence en boucle fermée estimée sur les PRBS longues (sans bruit)")
    figB.tight_layout(); figB.savefig(SORTIE_LONG / "prbs_long_bode.png", dpi=110)
    json.dump(resultats, open(SORTIE_LONG / "prbs_long_resultats.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    np.savez_compressed(SORTIE_LONG / "prbs_long_trajectoires.npz", **npz)

    print(f"{'essai':<8}{'contrôleur':<44}{'durée':>7}{'RMSE':>9}{'1er/dern. quart':>17}{'écart max':>10}"
          f"{'retard':>8}{'bande -3 dB':>13}{'agitation':>10}")
    for cle in ESSAIS_LONG:
        for lab, r in resultats[cle].items():
            bp = (f"{r['bande_passante_hz']:.2f} Hz" if r["bande_superieure_a_hz"] is None
                  else f"> {r['bande_superieure_a_hz']:.2f} Hz")
            print(f"{cle:<8}{lab:<44}{r['duree_s']:>6.0f}s{r['rmse']:>9.4f}{r['rmse_1er_quart']:>9.4f}/"
                  f"{r['rmse_dernier_quart']:<7.4f}{r['e_max']:>10.3f}{r['retard_s']:>7.2f}s{bp:>13}{r['agitation']:>10.4f}")
    print(f"\nÉcrit dans {SORTIE_LONG}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--court", action="store_true", help="essais courts (sinon : essais longs)")
    p.add_argument("--x", default=None, help="essais courts : dossier d'un autre agent x")
    a = p.parse_args()
    main_court(a.x) if a.court else main_long()
