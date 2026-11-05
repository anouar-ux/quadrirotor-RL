"""
Identification du TANGAGE (θ) et du ROULIS (φ), étape C1 : commande différentielle -> accélération angulaire.
Données : ../donnees/attitude/donnees_attitude.mat (générées par generation_donnees/generation_donnees_attitude.m).

    python identification_attitude.py

Le lacet a montré qu'un ARX(2,6) fait aussi bien que le GRU (dynamique linéaire : double intégrateur + retard moteur).
On identifie donc directement des ARX, avec les fonctions de identification_psi.py :
    θ : u_theta = (u3 - u1) / 2  ->  wdy        φ : u_phi = (u2 - u4) / 2  ->  wdx
On vérifie aussi :
  - le COUPLAGE entre axes (u_phi explique-t-il une part de wdy, et inversement ?) ;
  - la translation pour l'étape C2 : ax = c * theta + d * vx (gravité inclinée, frottement éventuel), ay = c * phi + d * vy.
    theta est l'intégrale de wy : la sortie theta de UAV_model est inutilisable (bruit de Cov = seed_theta).
Sorties (modeles/attitude/) : arx_attitude.npz, results.json, attitude.png.
"""
import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import PYTHON  # noqa: E402  (rend importables les sous-dossiers de python/)
HERE = PYTHON                # racine python/ : modeles/, resultats/ ; HERE.parent = EI_RL/
from identification_psi import arx_fit, arx_simulate, bilan, split_trials

DATA = HERE.parent / "donnees" / "attitude" / "donnees_attitude.mat"
OUT = HERE / "modeles" / "attitude"
TS = 0.05
NA, NB = 2, 6
AXES = {"theta": ("u_theta", "wdy", "u_phi"), "phi": ("u_phi", "wdx", "u_theta")}   # (commande, sortie, autre commande)
TRANSLATION = {"x": ("ax", "theta", "vx"), "y": ("ay", "phi", "vy")}


def reponse_echelon(a, b, u=0.003, n=40):
    """Accélération angulaire de l'ARX pour un échelon de commande u (comme une impulsion de l'étape A, sans fin)."""
    return arx_simulate(a, b, np.full((1, n), u))[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = loadmat(DATA, squeeze_me=True)
    S = {k: np.atleast_2d(D[k]).astype(np.float64) for k in ("u_theta", "u_phi", "wdy", "wdx", "wy", "wx", "theta", "phi",
                                                            "ax", "ay", "vx", "vy", "theta_mesure")}
    n = len(S["u_theta"])
    tr, va, te = split_trials(n, (0.7, 0.15, 0.15), 0)
    tr = np.r_[tr, va]                                  # pas d'hyperparamètre à régler : validation + entraînement
    print(f"{n} essais x {S['u_theta'].shape[1]} pas ; entraînement {len(tr)}, test {len(te)}")
    print(f"sortie theta de UAV_model : écart-type {S['theta_mesure'].std():.1f} rad (inutilisable) ; "
          f"theta estimé (intégrale de wy) : {S['theta'].std():.3f} rad")

    res, modeles = {}, {}
    for nom, (cu, cy, cautre) in AXES.items():
        U, Y = S[cu], S[cy]
        a, b = arx_fit(U[tr], Y[tr], NA, NB)
        Yh = arx_simulate(a, b, U[te])
        r = bilan(Yh, Y[te])
        gain = float(b.sum() / (1 - a.sum()))
        # couplage : régression du résidu de l'ARX sur l'AUTRE commande (mêmes retards)
        err = Y[te] - Yh
        rows = np.stack([S[cautre][te][:, NB - j:S[cautre].shape[1] - j] for j in range(1, NB + 1)], -1).reshape(-1, NB)
        bc, *_ = np.linalg.lstsq(rows, err[:, NB:].reshape(-1), rcond=None)
        r.update(gain=gain, a=a.tolist(), b=b.tolist(), couplage_gain=float(bc.sum()),
                 bruit_hf=float(np.std(np.diff(Y[te], axis=1)) / np.sqrt(2)), ecart_type_sortie=float(Y[te].std()))
        res[nom], modeles[nom] = r, (a, b)
        print(f"{nom:5s} : {cu} -> {cy}, ARX({NA},{NB}) a = {np.round(a, 4)}, b = {np.round(b, 2)}")
        print(f"        gain statique {gain:.1f} rad/s² par unité ; RMSE test {r['rmse']:.4f} rad/s² "
              f"(écart-type de {cy} : {r['ecart_type_sortie']:.3f}, bruit {r['bruit_hf']:.4f}) ; "
              f"dérive en 10 s : vitesse {r['derive_wz_10s']:.4f} rad/s, angle {r['derive_psi_10s']:.4f} rad")
        print(f"        couplage : {cautre} -> {cy} gain {bc.sum():+.2f} rad/s² par unité ({abs(bc.sum()) / gain:.2%} du gain direct)")

    for nom, (ca, cang, cv) in TRANSLATION.items():
        A, ANG, V = S[ca][tr].ravel(), S[cang][tr].ravel(), S[cv][tr].ravel()
        X = np.c_[ANG, V, np.ones_like(ANG)]
        c, *_ = np.linalg.lstsq(X, A, rcond=None)
        resid = A - X @ c
        res[f"translation_{nom}"] = dict(gain_angle=float(c[0]), frottement=float(c[1]), biais=float(c[2]), residu=float(resid.std()))
        print(f"{nom} : {ca} = {c[0]:+.3f} * {cang} {c[1]:+.4f} * {cv} {c[2]:+.3f}  (résidu {resid.std():.3f} m/s²)")

    np.savez(OUT / "arx_attitude.npz", **{f"{k}_{p}": v for k, (a, b) in modeles.items() for p, v in (("a", a), ("b", b))})
    json.dump(res, open(OUT / "results.json", "w"), indent=2)

    # Figures : un essai de test (mesure vs ARX en simulation libre, et angle intégré) + réponses à un échelon
    i, t = te[0], np.arange(S["u_theta"].shape[1]) * TS
    fig, ax = plt.subplots(3, 2, figsize=(13, 9), sharex="col")
    for c, (nom, (cu, cy, _)) in enumerate(AXES.items()):
        a, b = modeles[nom]
        Yh = arx_simulate(a, b, S[cu][[i]])[0]
        ax[0, c].plot(t, S[cu][i]); ax[0, c].set_ylabel(f"{cu} (appliquée)"); ax[0, c].set_title(f"{nom} : essai de test {i}")
        ax[1, c].plot(t, S[cy][i], lw=0.8, label="mesure"); ax[1, c].plot(t, Yh, "--", lw=1, label="ARX (simulation libre)")
        ax[1, c].set_ylabel(f"{cy} [rad/s²]"); ax[1, c].legend(fontsize=8)
        w0 = S["wy" if nom == "theta" else "wx"][i, 0]; ang0 = S[nom][i, 0]
        wh = w0 + np.cumsum(Yh) * TS
        ax[2, c].plot(t, S[nom][i], label=f"{nom} ({'intégrale de wy' if nom == 'theta' else 'mesure'})")
        ax[2, c].plot(t, ang0 + np.cumsum(wh) * TS, "--", label="ARX intégré deux fois")
        ax[2, c].set_ylabel(f"{nom} [rad]"); ax[2, c].set_xlabel("t depuis le début de l'excitation [s]"); ax[2, c].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "attitude.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for nom, (a, b) in modeles.items():
        ax.plot(np.arange(40) * TS, reponse_echelon(a, b) / 0.003, label=f"{nom} (gain {res[nom]['gain']:.0f})")
    ax.set_xlabel("t [s]"); ax.set_ylabel("accélération angulaire / u  [rad/s² par unité]")
    ax.set_title("Réponse de l'ARX à un échelon de commande : retard moteur"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "echelons.png", dpi=130); plt.close(fig)
    print(f"Écrit dans {OUT}")


if __name__ == "__main__":
    main()
