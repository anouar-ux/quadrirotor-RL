"""
TESTS DE L'ENVIRONNEMENT z (environnements/env_z.py), étape 3a : la physique et la consigne.
  1. chute libre (u = 0 pendant 1 s) : z, vz, az conformes à g ;
  2. commande d'équilibre u = 0,39 pendant 10 s : nombre de pas, dérive ;
  3. consigne aléatoire reproductible (même graine) et dans la plage [-0,5 ; 1] m ;
  4. consigne imposée. Lancer : python python/tests/test_env_z_3a.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chemins  # noqa: E402,F401  (sous-dossiers de python/)
import numpy as np
import matplotlib.pyplot as plt
from env_z import DroneEnv, TS, N_STEPS, Z_REF_RANGE

# Test 1 : chute libre, u = 0 pendant 1 s (20 pas)
env = DroneEnv()
env.reset(seed=0)
for _ in range(20):
    env.step(np.array([0.0]))
print(f"Test 1 (chute libre 1 s) : z = {env.z:.3f} m, vz = {env.vz:.3f} m/s, az = {env.az:.3f} m/s²")
print(f"   attendu (g pur)       : z = {-9.81 / 2:.3f} m, vz = {-9.81:.3f} m/s, az = -9.810 m/s²")

# Test 2 : commande d'équilibre u = 0,39 pendant 10 s, en notant z, vz, az à chaque pas
env.reset(seed=0)
zs, vzs, azs = [env.z], [env.vz], [env.az]
n_pas, truncated = 0, False
while not truncated:
    _, _, terminated, truncated, _ = env.step(np.array([0.39]))
    zs.append(env.z); vzs.append(env.vz); azs.append(env.az)
    n_pas += 1
print(f"Test 2 (u = 0.39) : {n_pas} pas (attendu {N_STEPS}), "
      f"z final = {zs[-1]:.3f} m, vz final = {vzs[-1]:.3f} m/s, az final = {azs[-1]:.3f} m/s²")

t = np.arange(n_pas + 1) * TS
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
for ax, y, nom in zip(axes, (zs, vzs, azs), ("z [m]", "vz [m/s]", "az [m/s²]")):
    ax.plot(t, y)
    ax.set_ylabel(nom)
    ax.grid(True)
axes[-1].set_xlabel("t [s]")
axes[0].set_title("Test 2 : u = 0.39 constant pendant 10 s")
fig.tight_layout()

# Test 3 : la consigne est reproductible et dans la bonne plage
env = DroneEnv()
env.reset(seed=0); a = env.z_ref
env.reset(seed=0); b = env.z_ref
print(f"Test 3a (reproductibilité) : {a:.6f} == {b:.6f} -> {a == b}")

refs = []
for _ in range(1000):
    env.reset()
    refs.append(env.z_ref)
print(f"Test 3b (1000 tirages) : min = {min(refs):.3f}, max = {max(refs):.3f} "
      f"(plage attendue {Z_REF_RANGE})")

# Test 4 : la consigne imposée fonctionne
env = DroneEnv(z_ref_fixe=0.5)
env.reset(seed=0)
print(f"Test 4 (consigne imposée) : z_ref = {env.z_ref} -> {env.z_ref == 0.5}")

plt.show()
