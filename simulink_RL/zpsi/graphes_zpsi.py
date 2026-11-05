"""Figures de la tâche avancée (z + lacet sur Simulink) à partir de resultats_zpsi.mat (vols_zpsi.m)."""
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.io import loadmat
ICI = Path(__file__).resolve().parent
R = loadmat(ICI / "resultats_zpsi.mat", squeeze_me=True, struct_as_record=False)["R"]
Tz, Tp = loadmat(ICI / "traj_z.mat", squeeze_me=True), loadmat(ICI / "traj_psi.mat", squeeze_me=True)
L0, LB = R[0].L, [r.L for r in R if r.g > 0]
t = L0[:, 0]
fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True, gridspec_kw=dict(height_ratios=[2, 2, 1.3, 1.3]))
ax[0].plot(t, L0[:, 23], "k--", lw=1.2, label="consigne lissée")
for j, L in enumerate(LB): ax[0].plot(t, L[:, 3], color="C1", lw=.8, alpha=.6, label="Simulink, g_noise = 0,1 (5 vols)" if j == 0 else None)
ax[0].plot(t, L0[:, 3], "C0", lw=1.6, label="Simulink, sans bruit"); ax[0].plot(Tz["t"], Tz["z"], color="gray", lw=4, alpha=.35, label="Python (GRU)")
ax[0].axhline(-0.25, color="brown", lw=1, label="sol"); ax[0].set_ylabel("z [m]"); ax[0].legend(fontsize=8, loc="upper right")
ax[1].plot(t, L0[:, 24], "k--", lw=1.2, label="consigne : un tour à vitesse constante")
for j, L in enumerate(LB): ax[1].plot(t, L[:, 9], color="C1", lw=.8, alpha=.6)
ax[1].plot(t, L0[:, 9], "C0", lw=1.6, label="Simulink"); ax[1].plot(Tp["t"], Tp["psi"], color="gray", lw=4, alpha=.35, label="Python (GRU)")
ax[1].axhline(2 * np.pi, color="k", lw=.5); ax[1].set_ylabel("psi [rad]"); ax[1].legend(fontsize=8, loc="lower right")
for j, L in enumerate(LB): ax[2].plot(t, np.degrees(L[:, 24] - L[:, 9]), color="C1", lw=.8, alpha=.6)
ax[2].plot(t, np.degrees(L0[:, 24] - L0[:, 9]), "C0", lw=1.4); ax[2].set_ylabel("erreur psi [°]")
for i in range(4): ax[3].step(t, L0[:, 19 + i], where="post", lw=1, label=f"u{i + 1}")
ax[3].set_ylabel("moteurs (sans bruit)"); ax[3].set_xlabel("t [s]"); ax[3].legend(fontsize=8, ncol=4)
for a in ax: a.grid(alpha=.3)
for a in ax[:3]: a.axvspan(5, 13, color="C2", alpha=.07)
ax[0].set_title("Agents PPO z + lacet sur UAV_model : décollage, tour complet (zone verte), atterrissage")
plt.tight_layout(); plt.savefig(ICI / "mission_zpsi_simulink.png", dpi=110)
for nom, Ls in (("sans bruit", [L0]), ("g_noise 0,1", LB)):
    f = lambda fn: np.mean([fn(L) for L in Ls])
    print(f"{nom:<12}: RMSE z {f(lambda L: np.sqrt(np.mean((L[:,23]-L[:,3])**2))):.4f} m | RMSE psi {f(lambda L: np.sqrt(np.mean((L[:,24]-L[:,9])**2))):.4f} rad "
          f"| erreur psi max {np.degrees(f(lambda L: np.max(np.abs(L[:,24]-L[:,9])))):.2f}° | pendant le scan (6-12 s) {np.degrees(f(lambda L: np.sqrt(np.mean((L[(t>6)&(t<12),24]-L[(t>6)&(t<12),9])**2)))):.2f}° "
          f"| z pendant le scan : moy {f(lambda L: L[(t>6)&(t<12),3].mean()):.4f}, é-t {100*f(lambda L: L[(t>6)&(t<12),3].std()):.2f} cm | vz au sol {f(lambda L: L[np.argmax((t>15)&(L[:,3]<-0.24)),6]):+.3f} m/s")
print("écart Simulink / Python sans bruit : z max %.4f m, psi max %.4f rad" % (np.max(np.abs(L0[:400,3]-Tz["z"])), np.max(np.abs(L0[:400,9]-Tp["psi"]))))
