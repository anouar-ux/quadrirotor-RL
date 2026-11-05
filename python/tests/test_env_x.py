"""Tests de l'environnement x (étape C2.1).
1. Boucle interne : on rejoue les consignes theta_ref du vol Simulink de C1 (test_LQ_ARX_attitude.m) dans BoucleTangage
   et on compare theta, x, vx simulés à ceux de Simulink (g_noise 0 et 0.1).
2. Consigne : v_ref est bien la dérivée de x_ref ; continuité ; cible qui roule.
3. API Gymnasium : formes, déterminisme avec la graine, action nulle -> drone immobile, ordre de grandeur des récompenses.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chemins  # noqa: E402,F401  (sous-dossiers de python/)
from env_z import TS  # noqa: E402
from env_x import (BoucleTangage, XEnv, charger_boucle_interne, consigne_x, mission_aleatoire,  # noqa: E402
                   N_STEPS)

RES_C1 = Path(__file__).resolve().parents[2] / "commande_LQ" / "LQ_ARX_attitude" / "resultats_test_LQ_ARX_attitude.mat"
ok = True


def verifie(cond, msg):
    global ok
    print(("  OK   " if cond else "  ECHEC ") + msg)
    ok &= bool(cond)


# ---------------------------------------------------------------- 1. boucle interne contre Simulink
print("1. Boucle interne (ARX + LQ + translation) contre le vol Simulink de C1")
V = sio.loadmat(RES_C1, squeeze_me=True, struct_as_record=False)["V"]
p = charger_boucle_interne("C1")         # le vol de C1 a été fait avec le gain K de gains_LQ_ARX_attitude.mat
for nom in ("g000", "g100"):
    L = getattr(V, nom)                       # colonnes : t, 18 sorties, u1..u4, theta_ref, phi_ref, theta_est
    t, x_s, vx_s, ax_s, th_ref, th_s = L[:, 0], L[:, 1], L[:, 4], L[:, 13], L[:, 23], L[:, 25]
    # dérive de theta_est sur Simulink : biais = theta_est - theta vrai, theta vrai = ax / G_X (lissé sur 1 s)
    biais = th_s - np.convolve(ax_s / p["g_x"], np.ones(20) / 20, "same")
    biais = np.convolve(biais, np.ones(20) / 20, "same")
    for avec_biais in (False, True):
        B = BoucleTangage(p)
        th_p, x_p, vx_p = [B.theta], [B.x], [B.vx]
        for k in range(len(t) - 1):
            bk = biais[k] if avec_biais else 0.0
            B.step(th_ref[k], bk)
            th_p.append(B.theta + (biais[k + 1] if avec_biais else 0.0)); x_p.append(B.x); vx_p.append(B.vx)
        th_p, x_p, vx_p = map(np.asarray, (th_p, x_p, vx_p))
        e_th = np.sqrt(np.mean((th_p - th_s) ** 2)); e_x = np.max(np.abs(x_p - x_s)); e_vx = np.max(np.abs(vx_p - vx_s))
        lab = "avec la dérive mesurée" if avec_biais else "sans dérive          "
        print(f"   {nom} {lab} : RMSE theta_est {e_th:.5f} rad | ecart max x {e_x:.4f} m (course {np.ptp(x_s):.3f} m)"
              f" | ecart max vx {e_vx:.4f} m/s")
        if nom == "g000" and not avec_biais:
            verifie(e_th < 0.002, "g000 : theta reproduit (RMSE < 0.002 rad)")
            verifie(e_x < 0.05 * np.ptp(x_s), "g000 : x reproduit à 5 % de la course près")
        if nom == "g100":
            verifie(e_th < 0.01, f"g100 {lab.strip()} : theta_est reproduit (RMSE < 0.01 rad)")
            if avec_biais:
                verifie(e_x < 0.05 * np.ptp(x_s), "g100 avec la dérive : x reproduit à 5 % de la course près")

# ---------------------------------------------------------------- 2. consigne
print("2. Consigne")
ev = ((1.0, "p", 2.0), (4.0, "v", 0.5), (9.0, "p", -1.0), (13.0, "v", 0.0))
tt = np.arange(0, N_STEPS + 1) * TS
X = np.array([consigne_x(ti, ev) for ti in tt])
e_trap = np.max(np.abs(np.diff(X[:, 0]) - TS / 2 * (X[1:, 1] + X[:-1, 1])))   # x(k+1) - x(k) = trapèze de v_ref
verifie(e_trap < 1e-3, f"v_ref = dérivée de x_ref (écart max par pas {e_trap:.2e} m)")
verifie(np.max(np.abs(np.diff(X[:, 0]))) < 0.1, "x_ref continue (pas de saut)")
verifie(abs(X[-1, 1]) < 1e-2, f"vitesse de cible ~ nulle à la fin ({X[-1, 1]:.4f} m/s)")
x_attendu = 2.0 - 1.0 + 0.5 * (13.0 - 4.0)   # le retard de lissage (2 TAU_X) du départ et de l'arrêt se compense
verifie(abs(X[-1, 0] - x_attendu) < 0.02, f"position finale {X[-1, 0]:.3f} m = 2 - 1 + 0.5 x 9 s = {x_attendu} m")
acc = np.gradient(X[:, 1], TS)
print(f"   accélération max de la consigne : {np.max(np.abs(acc)):.2f} m/s² (saut de 2 m, TAU_X = 1 s)")
rng = np.random.default_rng(0)
print("   exemples de missions aléatoires :")
for _ in range(3):
    print("    ", mission_aleatoire(rng))

# ---------------------------------------------------------------- 3. API
print("3. API Gymnasium")
env = XEnv()
o, _ = env.reset(seed=3)
verifie(o.shape == (7,) and env.observation_space.contains(o), "observation de 7 composantes")
o2, _ = XEnv().reset(seed=3)
verifie(np.array_equal(o, o2), "même graine -> même observation initiale")
env = XEnv(evenements_fixes=())
env.reset(seed=0)
R = 0.0
for _ in range(N_STEPS):
    o, r, term, trunc, info = env.step([0.0])
    R += r
verifie(abs(env.boucle.x) < 1e-9 and abs(env.boucle.theta) < 1e-9, "action nulle sans consigne : drone immobile")
verifie(trunc and abs(R - N_STEPS * 1.0) < 1e-6, f"vol parfait = +{N_STEPS} (obtenu {R:.1f})")
env = XEnv(evenements_fixes=((1.0, "p", 2.0),))
env.reset(seed=0)
R = sum(env.step([0.0])[1] for _ in range(N_STEPS))
print(f"   rester sur place face à un saut de 2 m : rendement {R:.1f}")
env = XEnv(evenements_fixes=((1.0, "p", 2.0),))
env.reset(seed=0)
for _ in range(60):
    env.step([1.0])
B = env.boucle
print(f"   theta_ref = +0.2 pendant 3 s : theta {B.theta:.3f} rad, vx {B.vx:.2f} m/s (9.9 x 0.2 x 3 = 5.9 sans "
      f"retard), x {B.x:.2f} m")
verifie(abs(B.theta - 0.2) < 0.01, "theta atteint sa consigne de 0.2 rad")
env = XEnv(evenements_fixes=(), derive=0.002)
D = []
for s in range(20):
    env.reset(seed=s)
    for _ in range(N_STEPS):
        env.step([0.0])
    D.append(env.boucle.x)
print(f"   dérive 0.002 rad/sqrt(s), action nulle, 20 s : |x| final moyen {np.mean(np.abs(D)):.2f} m "
      f"(max {np.max(np.abs(D)):.2f}) -> l'agent devra corriger en boucle fermée")
verifie(np.mean(np.abs(D)) > 0.05, "la dérive de theta_est fait bien dériver le drone")

print("\nTOUS LES TESTS PASSENT" if ok else "\nDES TESTS ECHOUENT")
sys.exit(0 if ok else 1)
