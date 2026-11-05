"""
TESTS DE L'ENVIRONNEMENT DU LACET (environnements/env_psi.py), étape B2.
  1. check_env de Stable-Baselines3 (sans et avec bruit) ;
  2. physique : une impulsion de commande donne la vitesse de lacet attendue (gain ~172 rad/s² par unité), sans frottement ;
  3. consigne : psi_ref est l'intégrale exacte de w_ref ; MISSION_SCAN fait un tour complet ;
  4. observation au pas 0 et réplique écrite à la main pas à pas ; récompense maximale = 1.
Comparaison finale au LQI de lacet (gains d'origine et / 2). Lancer : python python/tests/test_env_psi.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chemins  # noqa: E402,F401  (sous-dossiers de python/)
import numpy as np
from stable_baselines3.common.env_checker import check_env

from env_psi import MISSION_SCAN, N_STEPS, TS, U_PSI_MAX, W_SCAN, YawEnv, consigne_lacet, recompense

np.set_printoptions(precision=5, suppress=True)
BRUIT = dict(sigma_psi=0.005, sigma_wz=0.005, sigma_wdz=0.05)   # ~ g_noise = 0.1 (mesuré à l'étape B1)

# Test 1 : check_env
for nom, env in (("sans bruit", YawEnv()), ("avec bruit", YawEnv(**BRUIT))):
    check_env(env, warn=True)
    print(f"Test 1 : check_env {nom} -> OK")

# Test 2 : physique. a = 0.5 pendant 1 s (u_psi = 0.015), puis 0 : wz final ~ 172 * 0.015 * 1 s = 2.58 rad/s, et constant
env = YawEnv(sauts_fixes=())
env.reset(seed=0)
for k in range(60):
    env.step([0.5 if k < 20 else 0.0])
print(f"\nTest 2 : wz après l'impulsion = {env.wz:.3f} rad/s (attendu ~ {172 * U_PSI_MAX * 0.5:.2f}), "
      f"psi = {env.psi:.3f} rad")
w1 = env.wz
for k in range(40):
    env.step([0.0])
print(f"         2 s plus tard sans commande : wz = {env.wz:.3f} (attendu ~ identique : pas de frottement)")

# Test 3 : la consigne. psi_ref est l'intégrale exacte de w_ref ; un tour complet pour MISSION_SCAN
t = np.arange(0, N_STEPS + 1) * TS
P = np.array([consigne_lacet(x, MISSION_SCAN) for x in t])
num = np.r_[0, np.cumsum((P[1:, 1] + P[:-1, 1]) / 2) * TS]
print(f"\nTest 3 : écart max psi_ref / intégrale numérique de w_ref : {np.max(np.abs(P[:, 0] - num)):.1e} rad (attendu ~1e-4, erreur des trapèzes)")
print(f"         w_ref au milieu du scan = {consigne_lacet(9.0, MISSION_SCAN)[1]:.4f} (attendu {W_SCAN:.4f}) ; "
      f"psi_ref(20 s) = {P[-1, 0]:.5f} (attendu 2 pi = {2 * np.pi:.5f})")

# Test 4 : pas 0 et cohérence de l'observation (réplique écrite à la main, comme pour z)
env = YawEnv(sauts_fixes=MISSION_SCAN)
obs, info = env.reset(seed=0)
print(f"\nTest 4 : obs au pas 0 = {obs} (attendu 0)")
rng = np.random.default_rng(1)
int_e, ecart = 0.0, 0.0
for k in range(N_STEPS):
    obs, *_ , info = env.step([rng.uniform(-0.3, 0.3)])
    m = info["m"]
    e = m["psi_ref"] - m["psi"]
    if abs(e) < 0.1:
        int_e += e * TS
    int_e = min(max(int_e, -0.5), 0.5)
    w_ref = consigne_lacet(m["t"], MISSION_SCAN)[1]
    ref = np.array([e / 0.5, m["wz"], m["wdz_prev"] / 5, m["a_prev"], int_e / 0.5, w_ref - m["wz"]], np.float32)
    ecart = max(ecart, np.max(np.abs(ref - obs)))
print(f"         écart max obs / réplique à la main sur {N_STEPS} pas : {ecart:.1e} (attendu 0)")
print(f"         recompense(0, 0, 0) = {recompense(0, 0, 0):.3f} (attendu 1)")


# Test 5 : référence LQI (tes gains de ident_psi.m) sur la mission du scan, avec et sans bruit
def vol_lqi(gains, bruit, seed=0):
    kp, kd, ki = gains
    env = YawEnv(sauts_fixes=MISSION_SCAN, **(BRUIT if bruit else {}))
    obs, info = env.reset(seed=seed)
    integ, total, E, A = 0.0, 0.0, [], []
    for k in range(N_STEPS):
        m = info["m"]
        e = m["psi_ref"] - m["psi"]
        integ += e * TS
        a = float(np.clip((kp * e - kd * m["wz"] + ki * integ) / U_PSI_MAX, -1, 1))
        obs, r, _, _, info = env.step([a])
        total += r
        E.append(consigne_lacet((k + 1) * TS, MISSION_SCAN)[0] - env.psi); A.append(a)
    E, A = np.array(E), np.array(A)
    return total, np.sqrt(np.mean(E ** 2)), np.max(np.abs(E[int(14 / TS):])), np.mean(np.abs(np.diff(A)))


print("\nTest 5 : LQI de lacet sur le scan (rendement | RMSE | erreur max après 14 s | agitation a)")
for nom, g in (("tes gains", (0.252011, 0.124877, 0.097409)), ("gains / 2", (0.126, 0.0624, 0.0487))):
    for bruit in (False, True):
        tot, rmse, emax, agit = vol_lqi(g, bruit)
        print(f"   {nom:<10} {'bruit' if bruit else 'sans bruit':<11}: {tot:7.1f} | {rmse:.4f} rad | {emax:.4f} rad | {agit:.3f}")
