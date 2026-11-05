"""
TESTS DE L'ENVIRONNEMENT z, étape 3d : le LQI double intégrateur (commande_LQ/gains_LQI.m) piloté dans l'environnement.
  1. consigne fixe 0,5 m, sans bruit, avec et sans le filtre de consigne propre au LQI (tau = 1,2 s de test_LQI_z.slx) ;
  2. et 3. 20 consignes aléatoires, sans puis avec bruit : rendement, RMSE, dépassement, erreur finale.
Sert de référence classique pour l'agent RL. Figure : tests/test_env_z_3d_lqi.png. Lancer : python python/tests/test_env_z_3d_lqi.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chemins  # noqa: E402,F401  (sous-dossiers de python/)
from env_z import DroneEnv, TS, N_STEPS, mission

# Gains de gains_LQI.m (déjà multipliés par 0.8), et paramètres de commande_LQ/test_LQI_z.slx
KP, KD, KI = 1.86, 0.88, 0.79
U0 = 0.38                       # compensation de la gravité
TAU = 1.2                       # filtre de consigne de commande_LQ/test_LQI_z.slx [s], appliqué EN PLUS de la consigne lissée de l'env
BRUIT = dict(sigma_z=0.005, sigma_vz=0.005, sigma_az=0.005)   # ~ g_noise = 0.1


class LQI:
    def __init__(self, tau=TAU):
        self.a_f = np.exp(-TS / tau) if tau else 0.0   # filtre du 1er ordre discrétisé exactement (0 : pas de filtre)

    def reset(self, m):
        self.integ = 0.0
        self.z_f = m["z"]                   # le filtre part de l'altitude de départ (comme Simulink, sortie initiale 0)

    def __call__(self, m):
        self.z_f = self.a_f * self.z_f + (1 - self.a_f) * m["z_ref"]   # consigne filtrée (sans filtre : z_f = z_ref)
        e = self.z_f - m["z"]                                           # erreur sur la consigne filtrée, avec la MESURE de z
        self.integ += e * TS                                            # Euler, comme le Discrete-Time Integrator
        u = U0 + KP * e - KD * m["vz"] + KI * self.integ                # -KD : le terme dérivé s'oppose à la vitesse
        return float(np.clip(u, -1.0, 1.0))


def vol(env, ctrl, seed):
    obs, info = env.reset(seed=seed)
    ctrl.reset(info["m"])
    total, z, u_log, truncated = 0.0, [], [], False
    while not truncated:
        u = ctrl(info["m"])
        obs, r, _, truncated, info = env.step(np.array([u]))
        total += r; z.append(env.z); u_log.append(u)
    return total, np.array(z), np.array(u_log), env.z_ref


def indicateurs(z, z_cible, z_debut=0.0):
    """RMSE par rapport à la consigne LISSÉE (celle de la récompense) et à l'échelon BRUT, dépassement, erreur finale."""
    z_lisse = np.array([mission(k * TS, z_cible) for k in range(1, len(z) + 1)])   # consigne au même instant que z
    rmse_lisse = np.sqrt(np.mean((z_lisse - z) ** 2))
    rmse_brut = np.sqrt(np.mean((z_cible - z) ** 2))
    sens = np.sign(z_cible - z_debut) or 1.0
    depassement = max(0.0, np.max(sens * (z - z_cible)))          # dépassement au-delà de la hauteur cible [m]
    return rmse_lisse, rmse_brut, depassement, z_cible - z[-1]


t = np.arange(1, N_STEPS + 1) * TS

# Test 1 : consigne fixe 0.5 m, sans bruit, avec et sans filtre de consigne
fig, (ax_z, ax_u) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax_z.axhline(0.5, color="k", ls=":", lw=1, label="hauteur cible 0.5 m")
ax_z.plot(t, [mission(x, 0.5) for x in t], "k--", lw=1, label="consigne lissée (env)")
print("Test 1 : consigne 0.5 m, sans bruit")
for tau in (TAU, None):
    total, z, u, z_ref = vol(DroneEnv(z_ref_fixe=0.5), LQI(tau), seed=0)
    rmse, rmse_b, dep, e_fin = indicateurs(z, z_ref)
    nom = f"+ filtre tau = {tau} s" if tau else "consigne lissée seule"
    print(f"   {nom:<22}: rendement {total:8.1f} | RMSE lissée {rmse:.3f} m (brute {rmse_b:.3f}) | z min {z.min():+.3f} m | "
          f"dépassement {dep:.3f} m | erreur finale {e_fin:+.4f} m | u final {u[-1]:.4f}")
    ax_z.plot(t, z, label=nom)
    ax_u.step(t, u, where="post", label=nom)
ax_z.set_ylabel("z [m]"); ax_z.legend(); ax_z.grid(alpha=.3)
ax_z.set_title("LQI dans l'environnement Python (GRU), consigne 0.5 m")
ax_u.axhline(0.39, color="gray", ls=":", lw=1, label="u_eq du GRU (0.39)")
ax_u.set_ylabel("u"); ax_u.set_xlabel("t [s]"); ax_u.legend(); ax_u.grid(alpha=.3)
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent / "test_env_z_3d_lqi.png", dpi=120)

# Tests 2 et 3 : 20 consignes aléatoires (graines 0..19), sans puis avec bruit, avec puis sans le filtre propre au LQI
print("\nTests 2 et 3 : 20 consignes aléatoires dans [-0.5, 1] m (moyennes)")
print(f"   {'filtre LQI':<12}{'bruit':<8}{'rendement':>11}{'RMSE lissée':>13}{'RMSE brute':>12}{'dépassement':>13}{'|erreur finale|':>17}")
for tau in (TAU, None):
    for nom_bruit, bruit in (("non", {}), ("oui", BRUIT)):
        res = []
        for seed in range(20):
            total, z, u, z_ref = vol(DroneEnv(**bruit), LQI(tau), seed)
            res.append((total, *indicateurs(z, z_ref)))
        res = np.array(res)
        print(f"   {str(tau) + ' s' if tau else 'aucun':<12}{nom_bruit:<8}{res[:, 0].mean():>11.1f}{res[:, 1].mean():>12.3f}m"
              f"{res[:, 2].mean():>11.3f}m{res[:, 3].mean():>12.3f}m{np.abs(res[:, 4]).mean():>15.4f}m")

plt.show()
