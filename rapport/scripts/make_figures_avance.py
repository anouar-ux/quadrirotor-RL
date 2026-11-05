"""Figure de l'étape A (interface du drone), tracée à partir de generation_donnees/essais_interface/resultats_essais_interface.mat. Lecture seule.
Lancer : python rapport/scripts/make_figures_avance.py
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

# colonnes de chaque essai : t, 18 sorties nommées (x y z vx vy vz φ θ ψ ωx ωy ωz ax ay az ω̇x ω̇y ω̇z), 4 commandes, 18 de all_outputs
R = sio.loadmat(ROOT / "generation_donnees" / "essais_interface" / "resultats_essais_interface.mat", squeeze_me=True, struct_as_record=False)["R"]
PHI, THETA, PSI, WZ = 6, 7, 8, 11

lac = R.lacet
t, Y = lac[:, 0], lac[:, 1:19]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
ax[0].plot(t, Y[:, WZ], color="#1F3A5F", lw=2, label="ω$_z$ [rad/s]")
ax[0].set_xlabel("t [s]")
ax[0].set_ylabel("vitesse de lacet ω$_z$ [rad/s]", color="#1F3A5F")
ax2 = ax[0].twinx()
ax2.plot(t, Y[:, PSI], color="#B85C00", lw=2)
ax2.set_ylabel("angle de lacet ψ [rad]", color="#B85C00")
ax[0].axvspan(3, 4, color="0.9")
ax[0].set_title("Impulsion de lacet (+,−,+,−), 0,02 entre 3 et 4 s", fontsize=10)
ax[0].grid(alpha=0.25)

a, b = R.A, R.B
ax[1].plot(a[:, 0], a[:, 1 + PHI], "-", color="#B85C00", lw=2, label="φ, motif (+,+,−,−)")
ax[1].plot(a[:, 0], a[:, 1 + THETA], "-", color="#1F3A5F", lw=2, label="θ, motif (+,+,−,−)")
# le motif (+,−,−,+) donne φ = θ = −0,42 : on ne trace que φ, en pointillé épais, car θ se confond avec la courbe précédente
ax[1].plot(b[:, 0], b[:, 1 + PHI], ":", color="#B85C00", lw=3.5, zorder=5, label="φ, motif (+,−,−,+)  (θ identique)")
ax[1].set_xlabel("t [s]")
ax[1].set_ylabel("angle [rad]")
ax[1].set_title("Impulsions diagonales de 0,005 entre 3 et 3,3 s", fontsize=10)
ax[1].legend(fontsize=8, frameon=False)
ax[1].grid(alpha=0.25)
for a in (ax[0], ax[1], ax2):
    for s in ("top",):
        a.spines[s].set_visible(False)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "interface_essais.png", dpi=170)
print("psi final", float(Y[-1, PSI]), "wz final", float(Y[-1, WZ]))
