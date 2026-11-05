"""Réponse à un échelon de 0.5 m du LQI dans l'environnement Python (GRU), dans les conditions de commande_LQ/test_LQI_z.slx :
échelon à t = 0, filtre de consigne 1/(1.2 s + 1), u = 0.38 + Kp e - Kd vz + Ki int(e), saturation [-1, 1].
Le contrôleur lit les mesures et agit tous les Ts = 0.05 s (comme EINT_RL_closed_loop.slx)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chemins  # noqa: E402,F401  (sous-dossiers de python/)
from env_z import DroneEnv, N_STEPS, TS

KP, KD, KI = 1.86, 0.88, 0.79
U0, TAU, ECHELON = 0.38, 1.2, 0.5

env = DroneEnv(z_ref_fixe=ECHELON)      # l'env sert de drone ; la consigne est gérée ici, comme dans test_LQI_z.slx
_, info = env.reset(seed=0)
a_f = np.exp(-TS / TAU)
z_f, integ = 0.0, 0.0
log = {k: [] for k in ("t", "z", "vz", "az", "u", "z_f")}
log["t"].append(0.0); log["z"].append(env.z); log["vz"].append(env.vz); log["az"].append(env.az)
truncated = False
while not truncated:
    m = info["m"]
    z_f = a_f * z_f + (1 - a_f) * ECHELON                 # consigne filtrée (échelon brut en entrée)
    e = z_f - m["z"]
    integ += e * TS
    u = float(np.clip(U0 + KP * e - KD * m["vz"] + KI * integ, -1.0, 1.0))
    _, _, _, truncated, info = env.step(np.array([u]))
    log["u"].append(u); log["z_f"].append(z_f)
    log["t"].append(env.k * TS); log["z"].append(env.z); log["vz"].append(env.vz); log["az"].append(env.az)
L = {k: np.array(v) for k, v in log.items()}

# Fréquence et amortissement de l'oscillation, estimés sur les maxima de z après 3 s
z, t = L["z"], L["t"]
pics = [i for i in range(1, len(z) - 1) if z[i] > z[i - 1] and z[i] >= z[i + 1] and t[i] > 3]
if len(pics) >= 2:
    periode = np.mean(np.diff(t[pics]))
    rapport = (z[pics[-1]] - ECHELON) / (z[pics[0]] - ECHELON)
    print(f"Oscillation : période {periode:.2f} s ({1 / periode:.2f} Hz), amplitude divisée par "
          f"{1 / rapport:.1f} entre t = {t[pics[0]]:.1f} s et t = {t[pics[-1]]:.1f} s")
print(f"z max {z.max():.3f} m, z min {z.min():.3f} m, z final {z[-1]:.3f} m ; "
      f"u entre {L['u'].min():.2f} et {L['u'].max():.2f} ; saturé {np.mean(np.abs(L['u']) >= 0.999) * 100:.0f} % du temps")

fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
ax[0].plot(t, z, label="z (drone)")
ax[0].plot(t[1:], L["z_f"], "--", label="consigne filtrée (τ = 1.2 s)")
ax[0].axhline(ECHELON, color="k", ls=":", lw=1, label="échelon 0.5 m")
ax[0].set_ylabel("z [m]"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title("LQI (Kp = 1.86, Kd = 0.88, Ki = 0.79) — environnement Python, contrôleur à 20 Hz")
ax[1].step(t[1:], L["u"], where="post", label="u")
ax[1].axhline(0.39, color="gray", ls=":", lw=1, label="u_eq du GRU (0.39)")
ax[1].set_ylabel("u"); ax[1].legend(); ax[1].grid(alpha=.3)
ax[2].plot(t, L["az"], label="az")
ax[2].plot(t, L["vz"], label="vz")
ax[2].axhline(0, color="k", lw=.5)
ax[2].set_ylabel("az [m/s²], vz [m/s]"); ax[2].set_xlabel("t [s]"); ax[2].legend(); ax[2].grid(alpha=.3)
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent / "test_LQI_z_echantillonne.png", dpi=120)
plt.show()
