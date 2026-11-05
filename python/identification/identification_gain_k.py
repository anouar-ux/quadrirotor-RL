"""
IDENTIFICATION DES GAINS k DES MODÈLES « DOUBLE INTÉGRATEUR » utilisés par les LQI double intégrateur (commande_LQ/gains_LQI.m).

À quoi sert ce script : chaque axe est modélisé par  accélération = k * commande  (puis deux intégrations). Le gain k est
estimé par moindres carrés :
  - z   : à partir des données nettoyées (donnees/nettoyees/) : a_z + g = k * u, sur toutes les trajectoires ;
  - psi : à partir de mesures faites à la main sur Simulink (impulsion u_psi * dt  ->  pente de psi) ;
  - theta : idem pour le tangage.
Remplace les trois scripts d'origine k_identification, k_psi_id.py et k_theta.py (mêmes calculs, mêmes données).

    python python/identification/identification_gain_k.py            # les trois axes
    python python/identification/identification_gain_k.py --axe z    # un seul axe ; --figure pour tracer la régression
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier python/ (chemins.py)
from chemins import DONNEES  # noqa: E402

# Mesures faites à la main sur Simulink : (impulsion u * dt, pente de l'angle) — mesures d'origine
MESURES = {
    "psi": ([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2],
            [34.45, 68.87, 102.02, 135.96, 169.92, 199.6, 249.37, 299.13, 339.66]),
    "theta": ([0.02, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2],
              [1.8, 68.87, 102.02, 135.96, 169.92, 199.6, 249.37, 299.13, 339.66]),
}


def gain_z():
    """k tel que a_z + g = k * u, moindres carrés sur toutes les trajectoires nettoyées (comme gains_LQI.m)."""
    dossier = DONNEES / "nettoyees"
    U = np.loadtxt(dossier / "bdd_in_clean.csv.gz", delimiter=",")
    A = np.loadtxt(dossier / "bdd_out_clean.csv.gz", delimiter=",")
    print(f"z : {U.shape[0]} trajectoires de {U.shape[1]} pas")
    # compensation de la gravité si l'accélération démarre autour de -9,81 (drone au repos)
    A = np.where(np.abs(A[:, :10].mean(axis=1, keepdims=True) + 9.81) < 1.0, A + 9.81, A)
    k, *_ = np.linalg.lstsq(U.reshape(-1, 1), A.reshape(-1, 1), rcond=None)
    return float(k[0, 0])


def gain_mesures(axe, figure=False):
    """k = somme(x y) / somme(x²) : droite passant par l'origine (pente de l'angle = k * impulsion)."""
    x, y = (np.array(v, float) for v in MESURES[axe])
    k = float(np.sum(x * y) / np.sum(x ** 2))
    if figure:
        import matplotlib.pyplot as plt
        xl = np.linspace(0, x.max() * 1.1, 100)
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, color="tab:blue", label="points mesurés", zorder=5)
        plt.plot(xl, k * xl, color="tab:red", label=f"régression linéaire (k = {k:.2f})")
        plt.title(f"Estimation du gain k ({axe}) par régression linéaire")
        plt.xlabel(f"impulsion de commande u_{axe} · Δt"); plt.ylabel(f"pente de l'angle {axe}")
        plt.grid(True, linestyle="--", alpha=0.7); plt.legend(); plt.show()
    return k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--axe", choices=("z", "psi", "theta"), default=None)
    p.add_argument("--figure", action="store_true", help="trace la régression (axes psi et theta)")
    a = p.parse_args()
    for axe in ((a.axe,) if a.axe else ("z", "psi", "theta")):
        k = gain_z() if axe == "z" else gain_mesures(axe, a.figure)
        print(f"gain k_{axe} = {k:.4f}" + (" m/s² par unité de commande" if axe == "z" else " (pente / impulsion)"))


if __name__ == "__main__":
    main()
