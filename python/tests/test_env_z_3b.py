"""
TESTS DE L'ENVIRONNEMENT z, étape 3b : mesures et observation.
  1. pas 0 : info["m"] et observation (tout à 0, comme le UnitDelay de Simulink) ;
  2. pas 1 avec u = 0,39 : valeurs calculées à la main ;
  3. cohérence observation / mesures sur un vol ;
  4. intégration conditionnelle de l'erreur (seulement si |e| < 0,1 m). Lancer : python python/tests/test_env_z_3b.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chemins  # noqa: E402,F401  (sous-dossiers de python/)
import numpy as np

from env_z import DroneEnv, E_INT, I_MAX, N_STEPS, TAU_REF, TS, observation, vitesse_mission

np.set_printoptions(precision=5, suppress=True)
CLES = ("t", "z", "vz", "az_prev", "u_prev", "z_ref")   # ce qu'exige export_policy.py


def afficher_m(m):
    return ", ".join(f"{k} = {m[k]:.6g}" for k in CLES)


# Test 1 : reset(seed=0) -> info["m"] et observation du pas 0
env = DroneEnv()
obs, info = env.reset(seed=0)
m = info["m"]
print("Test 1 (pas 0)")
print(f"   info['m'] : {afficher_m(m)}")
print(f"   obs       : {obs}")
print(f"   attendu   : t = 0, z = 0, vz = 0, az_prev = 0, u_prev = 0, z_ref = 0 (consigne lissée : part de 0) ; "
      f"obs = [0. 0. 0. 0. 0. 0.]")
print(f"   clés présentes : {set(CLES) <= set(m)}")

# Test 2 : un pas avec u = 0.39 -> info["m"] et observation du pas 1
obs, _, _, _, info = env.step(np.array([0.39], dtype=np.float32))
m = info["m"]
print("\nTest 2 (pas 1, u = 0.39)")
print(f"   info['m'] : {afficher_m(m)}")
print(f"   obs       : {obs}")
print(f"   attendu   : t = 0.05, z = -0.010962, vz = -0.43850, az_prev = -9.7940, u_prev = 0.39, z_ref = 0.0042276 ; "
      f"obs = [0.03038 -0.4385 -0.9794 0.39 0.00152 0.59965]  (v_ref(0.05) = 0.16115 m/s)")

# Test 3 : l'observation se recalcule à partir de info["m"] seul (+ l'intégrale), à chaque pas d'un vol complet.
# On refait aussi l'intégrale "à la main", comme le fera rl_controller.m, à partir des seules mesures.
env = DroneEnv()
obs, info = env.reset(seed=1)
rng = np.random.default_rng(0)
int_matlab = 0.0
ecart_obs, ecart_int, n = 0.0, 0.0, 0
truncated = False
while True:
    m = info["m"]
    e = m["z_ref"] - m["z"]
    if abs(e) < E_INT:                                                   # intégration conditionnelle,
        int_matlab += e * TS                                             # comme le persistent de rl_controller.m
    int_matlab = min(max(int_matlab, -I_MAX), I_MAX)
    k = round(m["t"] / TS)
    v_ref = env.z_ref * (k * TS) / TAU_REF ** 2 * np.exp(-(k * TS) / TAU_REF)   # formule écrite à la main
    ecart_obs = max(ecart_obs, np.max(np.abs(observation(m, env.int_e, v_ref) - obs)))
    ecart_int = max(ecart_int, abs(int_matlab - env.int_e))
    n += 1
    if truncated:
        break
    obs, _, _, truncated, info = env.step(rng.uniform(-1, 1, size=1))                # commandes quelconques
print(f"\nTest 3 (cohérence sur {n} observations, commandes aléatoires)")
print(f"   écart max obs recalculée / obs renvoyée : {ecart_obs:.2e}  (attendu 0)")
print(f"   écart max intégrale 'MATLAB' / env      : {ecart_int:.2e}  (attendu 0)")
print(f"   {n} observations pour {N_STEPS} pas (attendu {N_STEPS + 1} : une par appel du contrôleur)")

# Test 4 : intégration conditionnelle. u = 0.39 : le drone tombe, l'erreur dépasse vite E_INT = 0.1 m ;
# l'intégrale ne doit plus bouger ensuite (pas d'emballement pendant les grands écarts)
env = DroneEnv()
obs, info = env.reset(seed=0)
truncated, hist = False, []
while not truncated:
    obs, _, _, truncated, info = env.step(np.array([0.39]))
    hist.append((abs(info["m"]["z_ref"] - info["m"]["z"]), env.int_e))
dernier_petit = max(i for i, (e, _) in enumerate(hist) if e < E_INT)
print("\nTest 4 (intégration conditionnelle)")
print(f"   |e| < {E_INT} m jusqu'au pas {dernier_petit + 1}, puis |e| final = {hist[-1][0]:.2f} m")
print(f"   int_e à ce moment = {hist[dernier_petit][1]:.5f}, int_e final = {env.int_e:.5f} (attendu identique, << {I_MAX})")
