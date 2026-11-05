"""
ANALYSE DES DONNÉES D'IDENTIFICATION DE z .

À quoi sert ce script : explorer les données nettoyées (donnees/nettoyees/) avant l'identification :
  1. forme des données (nombre d'essais, de pas) ;
  2. tracé d'un essai typique (u et az) ;
  3. retard : ajustement az(k) ~ a + b u(k - L) pour L = 0..6, sur tous les essais (le retard minimise la RMSE) ;
  4. bruit de mesure sur az (début des essais, u = 0).
    python python/identification/analyse_donnees_z.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DATA = Path(__file__).resolve().parents[2] / "donnees" / "nettoyees"
U = np.loadtxt(DATA / "bdd_in_clean.csv.gz", delimiter=",")
Y = np.loadtxt(DATA / "bdd_out_clean.csv.gz", delimiter=",")   # az [m/s²]
TS = 0.05

# 1. Forme
print(f"Nombre d'essais : {U.shape[0]}, Nombre de pas : {U.shape[1]}")

# 2. Un essai : tracer az et u (u × 10 pour qu'ils tiennent sur le même axe)
i = 0
t = np.arange(U.shape[1]) * TS
plt.figure()
plt.plot(t, U[i] * 10, label="u × 10")
plt.plot(t, Y[i], label="az")
plt.xlabel("temps [s]")
plt.ylabel("valeur")
plt.legend()
plt.title("Un essai typique")
plt.show()

# 3. Retard : pour L = 0..6, ajuster az(k) ≈ a + b·u(k−L) sur TOUS les essais
for L in range(7):
    x = U[:, :U.shape[1] - L].ravel()
    y = Y[:, L:].ravel()
    A = np.c_[np.ones_like(x), x]
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    rmse = np.sqrt(np.mean((A @ coef - y) ** 2))
    print(L, coef, rmse)

# 4. Bruit : écart-type de az au début des essais (u = 0, az stabilisé à -g)
# On retire la moyenne de chaque essai pour ne garder que les fluctuations.
debut = Y[:, 5:35]                     # pas 5 à 35 : u = 0, az stabilisé
bruit = np.std(debut - debut.mean(axis=1, keepdims=True))
print(f"Bruit de mesure sur az : {bruit:.4f} m/s²")

# 5. Caractéristique statique : az stabilisé en fin de palier, en fonction du niveau de u
us, azs = [], []
for i in range(300):
    sauts = np.r_[0, np.flatnonzero(np.diff(U[i]) != 0) + 1, U.shape[1]]
    for k0, k1 in zip(sauts[:-1], sauts[1:]):
        if k1 - k0 < 20:                   # 20 pas = 5 tau : az a le temps de se stabiliser
            continue
        us.append(U[i, k0])                    # u est constant sur le palier
        azs.append(np.mean(Y[i, k1-5:k1]))     # az moyen sur les 5 derniers pas du palier
us, azs = np.array(us), np.array(azs)
print(f"Paliers exploités : {len(us)}")


# 6. Trois modèles de la caractéristique statique, et l'équilibre u_eq (az stabilisé = 0) de chacun
def az_tanh(u, c, A, b):
    """Poussée en tanh : c = az à u = 0 (gravité), A = amplitude, b = raideur."""
    return c + A * np.tanh(b * u)


p1 = np.polyfit(us, azs, 1)                                   # droite
p3 = np.polyfit(us, azs, 3)                                   # polynôme de degré 3
(c, A, b), _ = curve_fit(az_tanh, us, azs, p0=(-9.81, 20.0, 1.0))   # tanh (moindres carrés non linéaires)


def racine_dans(p):
    """Racine réelle du polynôme p comprise dans [-1, 1]."""
    return [r.real for r in np.roots(p) if abs(r.imag) < 1e-9 and -1 <= r.real <= 1]


rmse = lambda pred: np.sqrt(np.mean((pred - azs) ** 2))
print(f"droite   : az = {p1[0]:.3f}·u {p1[1]:+.3f}"
      f"            RMSE {rmse(np.polyval(p1, us)):.3f} m/s², u_eq = {np.round(racine_dans(p1), 4)}")
print(f"degré 3  : coefficients {np.round(p3, 3)}"
      f"  RMSE {rmse(np.polyval(p3, us)):.3f} m/s², u_eq = {np.round(racine_dans(p3), 4)}")
# az = c + A·tanh(b·u) = 0  <=>  u = atanh(-c/A) / b   (formule exacte, possible car tanh est inversible)
u_eq_tanh = np.arctanh(-c / A) / b
print(f"tanh     : az = {c:.3f} + {A:.3f}·tanh({b:.4f}·u)"
      f"  RMSE {rmse(az_tanh(us, c, A, b)):.3f} m/s², u_eq = {u_eq_tanh:.4f}")

uu = np.linspace(-1, 1, 200)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(us, azs, ".", ms=2, color="gray", label="paliers mesurés")
ax1.plot(uu, np.polyval(p1, uu), "--", label="droite")
ax1.plot(uu, np.polyval(p3, uu), label="polynôme degré 3")
ax1.plot(uu, az_tanh(uu, c, A, b), ":", lw=2, label="tanh")
ax1.axhline(0, color="k", lw=0.5)
ax1.set_xlabel("u"); ax1.set_ylabel("az stabilisé [m/s²]"); ax1.legend()
ax1.set_title("Caractéristique statique")
# résidus : ce que chaque modèle n'explique pas (le bon modèle ne laisse que du bruit, sans forme)
ax2.plot(us, azs - np.polyval(p3, us), ".", ms=2, label="résidus degré 3")
ax2.plot(us, azs - az_tanh(us, c, A, b), ".", ms=2, label="résidus tanh")
ax2.axhline(0, color="k", lw=0.5)
ax2.set_xlabel("u"); ax2.set_ylabel("mesure - modèle [m/s²]"); ax2.legend()
ax2.set_title("Résidus")
fig.tight_layout()
plt.show()
