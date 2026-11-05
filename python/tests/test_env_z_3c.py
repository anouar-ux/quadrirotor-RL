"""
TESTS DE L'ENVIRONNEMENT z, étape 3c : récompense et bruit.
  1. check_env de Stable-Baselines3, sans puis avec bruit ;
  2. récompense du premier pas, comparée au calcul à la main ;
  3. récompense en fonction de l'écart de position, pénalités extrêmes (vitesse, douceur) ;
  4. le bruit ne touche que les mesures, jamais l'état physique ; bruit reproductible. Lancer : python python/tests/test_env_z_3c.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chemins  # noqa: E402,F401  (sous-dossiers de python/)
import numpy as np
from stable_baselines3.common.env_checker import check_env

from env_z import LAMBDA_DU, N_STEPS, TS, DroneEnv, mission, recompense, vitesse_mission

np.set_printoptions(precision=5, suppress=True)
BRUIT = dict(sigma_z=0.005, sigma_vz=0.005, sigma_az=0.005)   # ~ g_noise = 0.1 dans Simulink

# Test 1 : check_env de Stable-Baselines3, sans puis avec bruit
for nom, env in (("sans bruit", DroneEnv()), ("avec bruit", DroneEnv(**BRUIT))):
    check_env(env, warn=True)
    print(f"Test 1 : check_env {nom} -> OK")

# Test 2 : récompense du premier pas (reset(seed=0), u = 0.39)
env = DroneEnv()
env.reset(seed=0)
_, r, _, _, info = env.step(np.array([0.39], dtype=np.float32))
t = env.k * TS
e = mission(t, env.z_ref) - env.z                     # écart à la consigne LISSÉE à t = 0.05 s
e_v = vitesse_mission(t, env.z_ref) - env.vz          # écart à la vitesse de consigne
print(f"\nTest 2 : e = {e:.6f} m, e_v = {e_v:.5f} m/s, du = 0.39 -> r = {r:.5f}  (attendu ~ 0.29898, calculé à la main)")

# Test 3 : la récompense en fonction de l'écart de position seul (e_v = 0, du = 0), puis les pénalités
print("\nTest 3 : recompense(e, 0, 0)")
for x in (0, 0.01, 0.05, 0.1, 0.3, 1.0):
    print(f"   e = {x:4.2f} m -> {recompense(x, 0, 0):.5f}")
print("   attendu : 1 ; 0.97784 ; 0.66024 ; 0.43658 ; 0.12394 ; -0.19999")
print(f"   pénalités extrêmes : vitesse (e_v = 5 m/s) {recompense(0, 5, 0) - 1:.3f} (attendu -0.2) ; "
      f"commande (du = 0.2) {recompense(0, 0, 0.2) - 1:.3f} (attendu -0.12)")


# Test 4 : le bruit ne touche que les mesures. Même graine, mêmes actions, avec et sans bruit
def vol(env, actions, seed):
    obs, info = env.reset(seed=seed)
    z, ecarts, az0 = [], {"z": [], "vz": [], "az_prev": []}, info["m"]["az_prev"]
    for a in actions:
        obs, _, _, _, info = env.step(np.array([a]))
        z.append(env.z)
        m = info["m"]
        ecarts["z"].append(m["z"] - env.z)
        ecarts["vz"].append(m["vz"] - env.vz)
        ecarts["az_prev"].append(m["az_prev"] - env.az_prev)
    return np.array(z), {k: np.std(v) for k, v in ecarts.items()}, az0


actions = np.random.default_rng(0).uniform(-1, 1, N_STEPS)
# même consigne dans les deux vols : on l'impose, car les tirages de bruit décalent self.np_random
z_sans, std_sans, _ = vol(DroneEnv(z_ref_fixe=0.5), actions, seed=3)
z_avec, std_avec, az0_avec = vol(DroneEnv(z_ref_fixe=0.5, **BRUIT), actions, seed=3)
print("\nTest 4 : bruit")
print(f"   état physique identique avec / sans bruit : écart max {np.max(np.abs(z_avec - z_sans)):.1e} m (attendu 0)")
print(f"   écart-type mesure - vérité sans bruit : " + ", ".join(f"{k} {v:.4f}" for k, v in std_sans.items()))
print(f"   écart-type mesure - vérité avec bruit : " + ", ".join(f"{k} {v:.4f}" for k, v in std_avec.items())
      + "  (attendu ~0.005)")
print(f"   az_prev au pas 0 avec bruit : {az0_avec} (attendu 0.0)")
env_a, env_b = DroneEnv(**BRUIT), DroneEnv(**BRUIT)
oa, _ = env_a.reset(seed=7); ob, _ = env_b.reset(seed=7)
oa, *_ = env_a.step([0.4]); ob, *_ = env_b.step([0.4])
print(f"   bruit reproductible avec la même graine : {np.array_equal(oa, ob)}")

# Test 5 : rendement d'un vol à u = 0.39 constant (la valeur "à battre")
env = DroneEnv()
env.reset(seed=0)
total, truncated = 0.0, False
while not truncated:
    _, r, _, truncated, _ = env.step(np.array([0.39]))
    total += r
print(f"\nTest 5 : rendement à u = 0.39 constant = {total:.0f}  (z final {env.z:.2f} m, consigne {env.z_ref:.3f} m)")
