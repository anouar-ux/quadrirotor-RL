"""Figures de la cascade en x. Lancer : python rapport/scripts/make_figures_cascade.py
- cascade_x_gains.png : tracé du tableau de réglages (valeurs mesurées sur Simulink, saisies à la main) ;
- cascade_mission_voiture.png : résultats de commande_LQ/mission_LQ/resultats_mission_LQ.mat (lecture seule).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rapport" / "figures"
BLEU, ORANGE, GRIS = "#1F3A5F", "#B85C00", "0.5"

# réglages (Kp_x, Kd_x, ω_n, ζ, dépassement %, stabilisation 2 % en s, % du temps θ_ref saturé), g_noise = 0
T = [(0.10, 0.30, 0.99, 1.49, 0, 10.6, 0), (0.23, 0.28, 1.50, 0.91, 0.5, 3.6, 2),
     (0.41, 0.33, 2.01, 0.81, 6.5, 4.0, 6), (0.41, 0.26, 2.01, 0.64, 29, 6.7, 8), (0.64, 0.41, 2.51, 0.80, 29, 6.2, 12)]
lab = [f"{k:.2f} / {d:.2f}" for k, d, *_ in T]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
cols = [GRIS, BLEU, GRIS, GRIS, GRIS]
ax[0].bar(range(5), [r[5] for r in T], color=cols)
ax[0].set_ylabel("temps de stabilisation à 2 % [s]")
ax[1].bar(range(5), [r[4] for r in T], color=cols)
ax[1].set_ylabel("dépassement [%]")
for a in ax:
    a.set_xticks(range(5))
    a.set_xticklabels(lab, fontsize=8)
    a.set_xlabel("K$_{p,x}$ / K$_{d,x}$")
    a.grid(alpha=0.25, axis="y")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
ax[0].set_title("réglage retenu en bleu", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "cascade_x_gains.png", dpi=170)
plt.close(fig)

R = sio.loadmat(ROOT / "commande_LQ" / "mission_LQ" / "resultats_mission_LQ.mat", squeeze_me=True, struct_as_record=False)["R"]
S = R[2].S  # g_noise = 0,1
fig, ax = plt.subplots(2, 2, figsize=(10, 6.2))
ax[0, 0].plot(S.t, S.x_v, color=ORANGE, lw=2, label="voiture")
ax[0, 0].plot(S.t, S.x_ref, "--", color=GRIS, lw=1.2, label="x$_{ref}$")
ax[0, 0].plot(S.t, S.x, color=BLEU, lw=1.6, label="drone")
ax[0, 0].set_ylabel("x [m]")
ax[0, 0].legend(frameon=False, fontsize=8)
ax[0, 1].plot(S.t, S.x - S.x_v, color=BLEU)
ax[0, 1].set_ylabel("x drone − x voiture [m]")
ax[0, 1].set_ylim(-6, 0.5)
ax[1, 0].plot(S.t, S.z, color=BLEU)
ax[1, 0].set_ylabel("z [m]")
ax[1, 1].plot(S.t, S.psi, color=BLEU)
ax[1, 1].set_ylabel("ψ [rad]")
ax[1, 1].axhline(2 * np.pi, color=GRIS, lw=0.8, ls=":")
for a in ax.ravel():
    a.set_xlabel("t [s]")
    a.grid(alpha=0.25)
    for ph in range(1, 6):
        i = np.flatnonzero(S.phase == ph)
        if len(i) and ph > 1:
            a.axvline(S.t[i[0]], color="0.8", lw=0.8)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "cascade_mission_voiture.png", dpi=170)
