"""Figures et bilan chiffré des vols Simulink de l'agent z (mission de la Figure 1 du sujet), à partir de resultats_z.mat
(écrit par vols_z.m). Sorties : mission_z_simulink.png et mission_z_zooms.png, dans ce dossier."""
from pathlib import Path

import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.io import loadmat
ICI = Path(__file__).resolve().parent
R = loadmat(ICI / "resultats_z.mat", squeeze_me=True, struct_as_record=False)["R"]
T = loadmat(ICI.parents[1] / "rl_interface" / "traj_model.mat", squeeze_me=True)
t = R[0].L[:, 0]
echelon = np.where(t < 3, 0.0, np.where(t < 7, 0.5, -0.25))
G = {g: [r.L for r in R if abs(r.g - g) < 1e-9] for g in (0, 0.1, 0.2)}

def bilan(L):
    z, vz, u, zr = L[:, 1], L[:, 2], L[:, 4], L[:, 5]
    seg = lambda a, b: (t >= a) & (t < b)
    atter = t >= 7
    i = np.argmax(z[atter] < -0.24)
    return dict(rmse=np.sqrt(np.mean((zr - z) ** 2)), rmse_ech=np.sqrt(np.mean((echelon - z) ** 2)),
                p0=z[seg(2, 3)].mean(), p05=z[seg(5.5, 7)].mean(), p025=z[seg(9, 10.01)].mean(),
                dep=max(0, z[seg(3, 7)].max() - 0.5), vz_sol=vz[atter][max(i - 2, 0):i + 3].mean(),
                agit=np.mean(np.abs(np.diff(u))), sd_u=u[seg(5, 7)].std())
print(" g_noise | RMSE lissée | RMSE échelon | z moy 0 / 0,5 / -0,25 | dépass. | vz en arrivant au sol | agitation u | é-t u palier 0,5")
for g, Ls in G.items():
    B = [bilan(L) for L in Ls]; m = {k: np.mean([b[k] for b in B]) for k in B[0]}
    print(f"  {g:4.2f}   |   {m['rmse']:.4f}    |   {m['rmse_ech']:.3f}     | {m['p0']:+.3f} / {m['p05']:.3f} / {m['p025']:+.3f} |  {m['dep']:.3f}  |"
          f"      {m['vz_sol']:+.3f} m/s      |   {m['agit']:.3f}     |   {m['sd_u']:.3f}")

# ---- Figure 1 : la mission complète
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
ax[0].step(t, echelon, "k:", lw=1, where="post", label="paliers de la Figure 1")
ax[0].plot(t, R[0].L[:, 5], "k--", lw=1.2, label="consigne lissée (τ = 0,35 s)")
for j, L in enumerate(G[0.1]):
    ax[0].plot(t, L[:, 1], color="C1", lw=.8, alpha=.55, label="Simulink, g_noise = 0,1 (10 vols)" if j == 0 else None)
    ax[1].step(t, L[:, 4], color="C1", lw=.6, alpha=.45, where="post")
ax[0].plot(t, G[0][0][:, 1], color="C0", lw=1.8, label="Simulink, sans bruit")
ax[0].plot(T["t"], T["z"], color="gray", lw=4, alpha=.35, label="Python (modèle GRU), sans bruit")
ax[1].step(t, G[0][0][:, 4], color="C0", lw=1.4, where="post")
ax[0].axhline(-0.25, color="brown", lw=1, label="sol (-0,25 m)")
ax[0].set_ylabel("z [m]"); ax[1].set_ylabel("u"); ax[1].set_xlabel("t [s]")
ax[0].legend(fontsize=8, loc="upper right"); [a.grid(alpha=.3) for a in ax]
ax[0].set_title("Agent PPO sur UAV_model (Simulink) : mission de la Figure 1 avec bruit de mesure réaliste")
plt.tight_layout(); plt.savefig(ICI / "mission_z_simulink.png", dpi=110)

# ---- Figure 2 : zooms décollage et atterrissage (position et vitesse)
fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharex="col")
for c, (a, b, titre) in enumerate(((2.8, 5.5, "Montée à 0,5 m"), (6.8, 10, "Atterrissage à -0,25 m"))):
    k = (t >= a) & (t <= b)
    for j, L in enumerate(G[0.1]):
        ax[0, c].plot(t[k], L[k, 1], color="C1", lw=.8, alpha=.55, label="g_noise = 0,1" if j == 0 else None)
        ax[1, c].plot(t[k], L[k, 2], color="C1", lw=.8, alpha=.55)
    ax[0, c].plot(t[k], G[0][0][k, 1], color="C0", lw=1.8, label="sans bruit")
    ax[1, c].plot(t[k], G[0][0][k, 2], color="C0", lw=1.8)
    ax[0, c].plot(t[k], R[0].L[k, 5], "k--", lw=1.2, label="consigne lissée")
    # vitesse de consigne (dérivée numérique de la consigne lissée, pour l'affichage)
    ax[1, c].plot(t[k], np.gradient(R[0].L[:, 5], t)[k], "k--", lw=1.2, label="vitesse de consigne")
    ax[0, c].set_title(titre); ax[1, c].set_xlabel("t [s]")
    for r_ in (0, 1): ax[r_, c].grid(alpha=.3)
ax[0, 1].axhline(-0.25, color="brown", lw=1, label="sol")
ax[0, 0].set_ylabel("z [m]"); ax[1, 0].set_ylabel("vz [m/s]")
ax[0, 1].legend(fontsize=8); ax[1, 1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(ICI / "mission_z_zooms.png", dpi=110)
