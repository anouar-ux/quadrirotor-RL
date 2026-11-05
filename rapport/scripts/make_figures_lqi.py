"""Figures d'identification des gains de poussée (altitude et lacet). Lecture seule sur les données du projet.
Lancer : python rapport/scripts/make_figures_lqi.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rapport" / "figures"
VS = ROOT / "python"
G = 9.81


def fig_z():
    U = np.loadtxt(ROOT / "donnees" / "nettoyees" / "bdd_in_clean.csv", delimiter=",")
    Y = np.loadtxt(ROOT / "donnees" / "nettoyees" / "bdd_out_clean.csv", delimiter=",")
    # gain global : az + g = k·u par moindres carrés (même principe que LQI.m)
    k = float((U.ravel() @ (Y + G).ravel()) / (U.ravel() @ U.ravel()))
    # paliers stabilisés (même sélection que python/identification/analyse_donnees_z.py)
    us, azs = [], []
    for i in range(300):
        s = np.r_[0, np.flatnonzero(np.diff(U[i]) != 0) + 1, U.shape[1]]
        for k0, k1 in zip(s[:-1], s[1:]):
            if k1 - k0 < 20:
                continue
            us.append(U[i, k0])
            azs.append(Y[i, k1 - 5:k1].mean() + G)
    us, azs = np.array(us), np.array(azs)
    loc = np.abs(us - 0.39) < 0.3
    pente_loc = np.polyfit(us[loc], azs[loc], 1)[0]
    print(f"k global = {k:.2f} ; pente locale autour de u_eq = {pente_loc:.1f} ; {len(us)} paliers")

    uu = np.linspace(-1, 1, 100)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(us, azs, ".", ms=3, color="0.55", label="paliers stabilisés mesurés")
    ax.plot(uu, k * uu, color="#1F3A5F", lw=2, label=f"modèle linéaire  az + g = k·u,  k = {k:.1f} m/s²")
    ax.axhline(G, color="0.3", lw=0.8, ls=":")
    ax.text(-0.98, G + 0.8, "az + g = g  (vol stationnaire)", fontsize=9, color="0.3")
    ax.axvline(0.39, color="#B85C00", lw=0.9, ls="--")
    ax.text(0.36, -4, "u = 0,39\nvol stationnaire\nréel", fontsize=9, color="#B85C00", ha="right")
    ax.axvline(G / k, color="#1F3A5F", lw=0.9, ls="--")
    ax.text(G / k + 0.02, 3, f"u = {G / k:.2f}\nselon le\nmodèle linéaire", fontsize=9, color="#1F3A5F")
    ax.set_xlabel("commande u")
    ax.set_ylabel("accélération hors gravité  az + g  [m/s²]")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "lqi_z_identification.png", dpi=170)
    plt.close(fig)


def fig_psi():
    x = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0])
    y = np.array([34.45, 68.87, 102.02, 135.96, 169.92, 199.6, 249.37, 299.13, 339.66])
    k = float((x * y).sum() / (x * x).sum())
    print(f"k_psi = {k:.2f} rad/s² (unités SI, voir l'étape A du rapport)")
    xl = np.linspace(0, x.max() * 1.1, 50)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(x, y, "o", color="#B85C00", label="mesures (réponses indicielles)")
    ax.plot(xl, k * xl, color="#1F3A5F", lw=2, label=f"régression  y = k·x,  k = {k:.1f} rad/s²")
    ax.set_xlabel("impulsion de commande  u_ψ · Δt")
    ax.set_ylabel("vitesse de lacet atteinte  [rad/s]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "lqi_psi_identification.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_z()
    fig_psi()
