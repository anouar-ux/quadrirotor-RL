"""
RÉAJUSTEMENT DES GAINS DES LQI « DOUBLE INTÉGRATEUR » POUR UN CONTRÔLEUR ÉCHANTILLONNÉ À 20 Hz.

Pourquoi : les gains de commande_LQ/gains_LQI.m sont réglés et validés sur les modèles Simulink du LQI, où
le LQI est CONTINU (solveur à pas variable, commande recalculée à chaque pas du solveur, sans bloqueur d'ordre zéro).
Échantillonnés à Ts = 0,05 s, comme dans l'interface RL et dans toutes les comparaisons Python, les mêmes gains voient
en plus un retard d'environ un demi-pas (bloqueur) qui s'ajoute à celui des moteurs : z oscille, theta diverge.
Ce n'est pas le réglage du LQI qui est en cause, c'est son implantation à 20 Hz.

Méthode : on garde EXACTEMENT la même loi de commande (même structure LQI) :
    z     : u     = u0 + Kp (z_ref - z) - Kd vz + Ki * intégrale(z_ref - z)                  (u0 = 0,38)
    psi   : u_psi = Kp (psi_ref - psi) - Kd wz + Ki * intégrale(psi_ref - psi)               (|u_psi| <= 0,03)
    theta : u_th  = Kp (theta_ref - theta) - Kd wy + Ki * intégrale(theta_ref - theta)      (|u_th| <= 0,5)
et on cherche (Kp, Kd, Ki) tels que la réponse à 20 Hz sur le modèle identifié (GRU de z, GRU du lacet, ARX de
l'attitude), avec la MÊME consigne que sur Simulink, soit la plus proche possible de la réponse Simulink enregistrée
par commande_LQ/reponses_LQI_simulink.m (moindres carrés, Nelder-Mead sur le logarithme des gains, départ : gains
d'origine). Les gains réajustés servent ensuite de « LQI double intégrateur » dans evaluation/comparaisons_LQI_ARX_RL.py.

    python python/identification/reajuster_gains_LQI.py   ->  modeles/gains_LQI_20Hz.json,
                                                              resultats/comparaisons/reajustement_gains_LQI.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.io import loadmat  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import COMMANDE_LQ, MODELES, RESULTATS  # noqa: E402
from env_psi import U_PSI_MAX, YawEnv  # noqa: E402
from env_x import U_MAX_ATT, charger_boucle_interne  # noqa: E402
from env_z import TS, DroneEnv  # noqa: E402

FICHIER = MODELES / "gains_LQI_20Hz.json"
U0_Z = 0.38


def charger_simulink():
    S = loadmat(COMMANDE_LQ / "reponses_LQI_simulink.mat", squeeze_me=True, struct_as_record=False)
    return S["REP"], S["gains"]


def echantillonner(d, nom, n):
    """Signal Simulink aux instants k * TS (k = 0..n-1) et aux instants suivants (k + 1) * TS."""
    t = np.arange(n + 1) * TS
    return np.interp(t, d.t, getattr(d, nom))


# ------------------------------------------------------------------ simulations à 20 Hz sur les modèles identifiés
def vol_z(gains, ref):
    kp, kd, ki = gains
    env = DroneEnv(z_ref_fixe=0.0); obs, info = env.reset(seed=0)
    I, Z = 0.0, []
    for k in range(len(ref) - 1):
        m = info["m"]; e = ref[k] - m["z"]; I += e * TS
        u = float(np.clip(U0_Z + kp * e - kd * m["vz"] + ki * I, -1.0, 1.0))
        obs, *_, info = env.step(np.array([u], np.float32)); Z.append(env.z)
    return np.array(Z)


def vol_psi(gains, ref):
    kp, kd, ki = gains
    env = YawEnv(sauts_fixes=((0.0, 0.0),)); obs, info = env.reset(seed=0)
    I, P = 0.0, []
    for k in range(len(ref) - 1):
        m = info["m"]; e = ref[k] - m["psi"]; I += e * TS
        a = float(np.clip((kp * e - kd * m["wz"] + ki * I) / U_PSI_MAX, -1.0, 1.0))
        obs, *_, info = env.step([a]); P.append(env.psi)
    return np.array(P)


def vol_theta(gains, ref):
    kp, kd, ki = gains
    p = charger_boucle_interne("rapide"); a, b = p["a"], p["b"]
    th = w = wd = wd1 = I = 0.0; U = np.zeros(5); T = []
    for k in range(len(ref) - 1):
        e = ref[k] - th; I += e * TS
        u = float(np.clip(kp * e - kd * w + ki * I, -U_MAX_ATT, U_MAX_ATT))
        wdn = a[0] * wd + a[1] * wd1 + b[0] * u + b[1:] @ U
        wn = w + TS / 2 * (wd + wdn); th = th + TS / 2 * (w + wn)
        U = np.r_[u, U[:4]]; wd1, wd, w = wd, wdn, wn; T.append(th)
        if abs(th) > 5:                                        # divergence : on arrête
            T += [th] * (len(ref) - 2 - k); break
    return np.array(T)


AXES = {"z": ("test_LQI_z", "z", "z_ref", vol_z, 200, "z [m]"),
        "psi": ("test_LQI_psi", "psi", "psi_ref", vol_psi, 400, "ψ [rad]"),
        "theta": ("test_LQI_x", "theta", "theta_ref", vol_theta, 300, "θ [rad]")}


def reajuster(axe, REP, g0):
    mdl, y, r, vol, n, _ = AXES[axe]
    d = getattr(REP, mdl)
    ref, cible = echantillonner(d, r, n), echantillonner(d, y, n)[1:]
    cout = lambda lg: float(np.sqrt(np.mean((np.clip(vol(np.exp(lg), ref), -10, 10) - cible) ** 2)))
    meilleurs = None
    for depart in (np.log(g0), np.log(np.asarray(g0) / 2), np.log(np.asarray(g0) / 4)):
        res = minimize(cout, depart, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6, maxiter=400))
        if meilleurs is None or res.fun < meilleurs.fun:
            meilleurs = res
    g = np.exp(meilleurs.x)
    return dict(gains_origine=[float(v) for v in g0], gains_20Hz=[float(v) for v in g],
                ecart_origine_a_simulink=cout(np.log(g0)), ecart_20Hz_a_simulink=float(meilleurs.fun),
                amplitude_reponse_simulink=float(np.ptp(cible))), (d, ref, cible, vol)


def main():
    REP, G = charger_simulink()
    g0 = {"z": np.atleast_1d(G.z), "psi": np.atleast_1d(G.psi), "theta": np.atleast_1d(G.theta)}
    R, F = {}, {}
    for axe in AXES:
        R[axe], F[axe] = reajuster(axe, REP, g0[axe])
        r = R[axe]
        print(f"{axe:6s}: gains d'origine {np.round(r['gains_origine'], 4)} -> 20 Hz {np.round(r['gains_20Hz'], 4)} ; "
              f"écart RMS à Simulink {r['ecart_origine_a_simulink']:.4f} -> {r['ecart_20Hz_a_simulink']:.4f} "
              f"(amplitude de la réponse {r['amplitude_reponse_simulink']:.3f})")
    FICHIER.write_text(json.dumps(R, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for j, axe in enumerate(AXES):
        mdl, y, r, vol, n, lab = AXES[axe]; d, ref, cible, _ = F[axe]
        t = np.arange(1, n + 1) * TS
        ax[j].plot(d.t, getattr(d, r), "k:", lw=1, label="consigne")
        ax[j].plot(d.t, getattr(d, y), color="0.55", lw=4, alpha=0.6, label="Simulink, LQI continu (gains d'origine)")
        yo = vol(R[axe]["gains_origine"], ref); yn = vol(R[axe]["gains_20Hz"], ref)
        ax[j].plot(t, np.clip(yo, *(np.array([-0.5, 1.5]) * max(np.abs(cible).max(), 0.1))), color="tab:red", lw=1,
                   label="20 Hz, gains d'origine" + (" (diverge, tracé borné)" if np.abs(yo).max() > 2 * np.abs(cible).max() else ""))
        ax[j].plot(t, yn, color="tab:orange", lw=1.6, label="20 Hz, gains réajustés")
        ax[j].set_title(f"{axe} : écart RMS à Simulink {R[axe]['ecart_origine_a_simulink']:.3f} -> {R[axe]['ecart_20Hz_a_simulink']:.3f}")
        ax[j].set_xlabel("t [s]"); ax[j].set_ylabel(lab); ax[j].grid(alpha=.3); ax[j].legend(fontsize=7)
    fig.suptitle("LQI double intégrateur : réponse Simulink (continu) et contrôleur échantillonné à 20 Hz, avant / après réajustement des gains")
    fig.tight_layout(); (RESULTATS / "comparaisons").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTATS / "comparaisons" / "reajustement_gains_LQI.png", dpi=115)
    print(f"-> {FICHIER}")


def gains_20Hz(axe):
    """Gains réajustés (Kp, Kd, Ki) d'un axe ('z', 'psi', 'theta'), lus dans modeles/gains_LQI_20Hz.json."""
    return tuple(json.loads(FICHIER.read_text(encoding="utf-8"))[axe]["gains_20Hz"])


if __name__ == "__main__":
    main()
