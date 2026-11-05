"""
SCHÉMA D'ARCHITECTURE du contrôleur de la mission finale (figure de la présentation, mode clair).

En haut : les 6 phases du superviseur et leurs conditions de passage (décidées sur les MESURES).
En bas : la chaîne de commande exécutée à chaque pas (20 Hz), de rl_controller_mission.m / sim_mission.py :
    mesures (18 sorties de UAV_model) + position de la voiture -> superviseur -> consignes
    -> agent z, agent lacet, rotation de -psi puis agent x sur les deux axes du drone -> theta_ref, phi_ref
    -> LQ d'attitude (modèle ARX) -> anticipation 1 / cos de la poussée -> mélangeur -> 4 moteurs.

    python python/presentation/architecture.py     ->  rapport/presentation/08_architecture.png (et .svg)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import commun as C  # noqa: E402

COUL = dict(rl="#cfe2f3", lq="#d9ead3", sup="#fff2cc", drone="#eeeeee", calc="#f4f4f4", voit="#fce5cd")


def boite(ax, x, y, l, h, texte, fc, fs=9, gras=False, ec="0.25"):
    ax.add_patch(FancyBboxPatch((x - l / 2, y - h / 2), l, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=1.2))
    ax.text(x, y, texte, ha="center", va="center", fontsize=fs, weight="bold" if gras else "normal", linespacing=1.25)


def fleche(ax, p, q, texte="", dx=0, dy=0.018, coul="0.2", style="-|>", ls="-", rad=0.0, fs=7.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=coul, lw=1.2, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if texte:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, texte, ha="center", va="bottom", fontsize=fs, color="0.15")


def main():
    C.style_clair()
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, 1.6); ax.set_ylim(0, 0.9); ax.axis("off")
    ax.text(0.8, 0.875, "Architecture du contrôleur de la mission finale (exécuté à 20 Hz, sur Simulink comme en Python)",
            ha="center", va="top", fontsize=15, weight="bold")

    # ---------------- superviseur : 6 phases
    ax.text(0.03, 0.80, "Superviseur : 6 phases, transitions décidées sur les mesures", fontsize=11, weight="bold", va="center")
    phases = [("0  stabilisation", "z = 0, 2 s"), ("1  montée", "z_ref : 0 → 0,5 m"), ("2  scan", "un tour de ψ en 10 s"),
              ("3  poursuite", "p_ref → voiture (τ = 1 s)\nψ : vers la voiture"), ("4  descente", "0,1 m/s, pause si écart > 10 cm"),
              ("5  posé", "suivi maintenu")]
    cond = ["2 s", "|z - 0,5| < 2 cm\n|vz| < 5 cm/s\npendant 1 s", "10 s écoulées\nψ ≈ 2π\nωz ≈ 0", "écart < 5 cm\n< 10 cm/s\ncap ± 3°\npendant 2 s",
            "z < 2 cm"]
    x0, pas = 0.14, 0.264
    for i, (nom, det) in enumerate(phases):
        x = x0 + i * pas
        boite(ax, x, 0.71, 0.165, 0.09, f"{nom}\n{det}", COUL["sup"], fs=8)
        if i < 5:
            fleche(ax, (x + 0.0825, 0.735), (x + pas - 0.0825, 0.735))
            ax.text(x + pas / 2, 0.722, cond[i], ha="center", va="top", fontsize=6.8, color="tab:red", linespacing=1.1)

    # ---------------- chaîne de commande
    ax.text(0.03, 0.585, "Chaîne de commande (à chaque pas de 50 ms)", fontsize=11, weight="bold", va="center")
    ax.plot([0.02, 1.58], [0.615, 0.615], color="0.75", lw=0.8)
    # entrées
    boite(ax, 0.11, 0.40, 0.16, 0.13, "Drone\n(UAV_model)\n18 mesures bruitées :\nx y z, vitesses,\nφ θ ψ, ω, accélérations", COUL["drone"], fs=8)
    boite(ax, 0.11, 0.17, 0.16, 0.08, "Voiture\nposition, vitesse, cap", COUL["voit"], fs=8.5)
    boite(ax, 0.36, 0.30, 0.17, 0.20, "Superviseur\n\nphase en cours\n→ consignes :\nz_ref, vz_ref\nψ_ref, ω_ref\np_ref, v_ref (x, y)",
          COUL["sup"], fs=8.5, gras=False)
    fleche(ax, (0.19, 0.40), (0.275, 0.34), "mesures", dy=0.012)
    fleche(ax, (0.19, 0.17), (0.275, 0.25), "", dy=0.01)
    # agents
    boite(ax, 0.66, 0.50, 0.19, 0.075, "Agent z (PPO)\n6 entrées → u_z", COUL["rl"], fs=9, gras=True)
    boite(ax, 0.66, 0.39, 0.19, 0.075, "Agent lacet (PPO)\n6 entrées → u_ψ", COUL["rl"], fs=9, gras=True)
    boite(ax, 0.565, 0.20, 0.12, 0.10, "Rotation de -ψ\nerreurs dans le\nrepère du drone\n(avant / côté)", COUL["calc"], fs=8)
    boite(ax, 0.76, 0.245, 0.17, 0.07, "Agent x (PPO), axe avant\n7 entrées → θ_ref", COUL["rl"], fs=8.5, gras=True)
    boite(ax, 0.76, 0.145, 0.17, 0.07, "même agent x, axe de côté\n(-φ, -ωx) → φ_ref", COUL["rl"], fs=8.5, gras=True)
    fleche(ax, (0.445, 0.36), (0.565, 0.50), "erreur z, vz_ref - vz", dy=0.03, dx=-0.045)
    fleche(ax, (0.445, 0.33), (0.565, 0.39), "erreur ψ, ω_ref - ωz", dy=0.018, dx=-0.01)
    fleche(ax, (0.445, 0.26), (0.505, 0.21))
    ax.text(0.47, 0.135, "p_ref - p\nv_ref - v", ha="center", va="top", fontsize=7.5, color="0.15")
    fleche(ax, (0.625, 0.22), (0.675, 0.245)); fleche(ax, (0.625, 0.18), (0.675, 0.145))
    # attitude
    boite(ax, 1.01, 0.195, 0.17, 0.15, "LQ d'attitude\n(modèle ARX identifié,\nretard des moteurs inclus)\nθ_ref → u_θ\nφ_ref → u_φ",
          COUL["lq"], fs=8.5, gras=False)
    fleche(ax, (0.845, 0.245), (0.925, 0.215)); fleche(ax, (0.845, 0.145), (0.925, 0.175))
    boite(ax, 1.01, 0.50, 0.17, 0.075, "anticipation de poussée\nu_z + 0,39 (1 / cos θ cos φ - 1)", COUL["calc"], fs=8)
    fleche(ax, (0.755, 0.50), (0.925, 0.50), "u_z", dy=0.006)
    # mélangeur et moteurs
    boite(ax, 1.30, 0.34, 0.17, 0.25, "Mélangeur\n\nu1 = u_z + u_ψ - u_θ\nu2 = u_z - u_ψ + u_φ\nu3 = u_z + u_ψ + u_θ\nu4 = u_z - u_ψ - u_φ",
          COUL["calc"], fs=8.5)
    fleche(ax, (1.095, 0.50), (1.215, 0.42)); fleche(ax, (0.755, 0.39), (1.215, 0.36), "u_ψ", dy=0.004, dx=0.1)
    fleche(ax, (1.095, 0.195), (1.215, 0.27), "u_θ, u_φ", dy=0.0, dx=0.02)
    boite(ax, 1.50, 0.34, 0.11, 0.12, "4 moteurs\nu1 … u4", COUL["drone"], fs=9, gras=True)
    fleche(ax, (1.385, 0.34), (1.445, 0.34))
    # retour
    ax.plot([1.50, 1.50, 0.012, 0.012], [0.28, 0.06, 0.06, 0.40], color="0.45", lw=1.1, ls="--")
    fleche(ax, (0.012, 0.40), (0.03, 0.40), coul="0.45")
    ax.text(0.8, 0.067, "boucle fermée sur le drone (Simulink : UAV_model ; Python : modèles identifiés GRU + ARX)",
            ha="center", fontsize=8, color="0.35")
    # légende
    for i, (c, lab) in enumerate(((COUL["rl"], "réseau appris par renforcement (PPO)"),
                                  (COUL["lq"], "commande LQ sur modèle identifié"),
                                  (COUL["sup"], "logique de mission"), (COUL["calc"], "calcul fixe (géométrie, mélange)"))):
        ax.add_patch(FancyBboxPatch((0.98 + (i % 2) * 0.29, 0.585 - (i // 2) * 0.035), 0.025, 0.02, boxstyle="round,pad=0.002", fc=c, ec="0.3"))
        ax.text(1.01 + (i % 2) * 0.29, 0.595 - (i // 2) * 0.035, lab, fontsize=8, va="center")
    for ext in ("png", "svg"):
        fig.savefig(C.PRESENTATION / f"08_architecture.{ext}", dpi=130)
    plt.close(fig)
    print(f"-> {C.PRESENTATION / '08_architecture.png'}")


if __name__ == "__main__":
    main()
