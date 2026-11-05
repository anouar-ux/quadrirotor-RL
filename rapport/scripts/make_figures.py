"""Figures complémentaires du rapport (aucun fichier du projet n'est modifié : lecture seule).
Lancer : python rapport/scripts/make_figures.py
Sorties : rapport/figures/fig_donnees.png, fig_consigne.png, fig_recompense.png, fig_synthese.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rapport" / "figures"
VS = ROOT / "python" / "resultats"
sys.path.insert(0, str(ROOT / "python" / "environnements"))

TS = 0.05


# ---------------------------------------------------------------- 1. données brutes
def fig_donnees():
    u = np.loadtxt(ROOT / "donnees" / "nettoyees" / "bdd_in_clean.csv", delimiter=",")
    az = np.loadtxt(ROOT / "donnees" / "nettoyees" / "bdd_out_clean.csv", delimiter=",")
    print("bdd :", u.shape, "essais x pas")
    t = np.arange(u.shape[1]) * TS
    fig, ax = plt.subplots(3, 2, figsize=(11, 7), sharex="col")
    for j, k in enumerate([0, 1, 2]):
        ax[j, 0].step(t, u[k], where="post", color="C0")
        ax[j, 0].set_ylabel(f"u (essai {k})")
        ax[j, 1].plot(t, az[k], color="C1", lw=0.8)
        ax[j, 1].set_ylabel("az [m/s²]")
    ax[2, 0].set_xlabel("t [s]")
    ax[2, 1].set_xlabel("t [s]")
    ax[0, 0].set_title("Entrée : commande normalisée u")
    ax[0, 1].set_title("Sortie : accélération verticale mesurée az")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(f"Base de données d'identification : {u.shape[0]} essais de {u.shape[1]} pas (Ts = 50 ms)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_donnees.png", dpi=150)
    plt.close(fig)

    # bruit de mesure estimé sur le premier palier (u constant, avant le premier saut)
    n0 = 40
    print("écart-type de az sur les 40 premiers pas (repos) :", float(np.mean(np.std(az[:, :n0], axis=1))))


# ---------------------------------------------------------------- 2. consigne lissée
def fig_consigne():
    import env_z as de
    t = np.arange(0, 10, TS)
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for k, zc in enumerate([0.5, 1.0, -0.5]):
        ax[0, 0].plot(t, [de.mission(x, zc) for x in t], label=f"cible {zc:+.1f} m")
        ax[1, 0].plot(t, [de.vitesse_mission(x, zc) for x in t])
    ax[0, 0].set_title("Entraînement : saut unique lissé (τ = 0,35 s)")
    ax[0, 0].set_ylabel("z_ref [m]")
    ax[1, 0].set_ylabel("v_ref [m/s]")
    ax[0, 0].legend()
    ax[0, 1].step([p[0] for p in de.MISSION_FIG1] + [10], [p[1] for p in de.MISSION_FIG1] + [de.MISSION_FIG1[-1][1]],
                  where="post", color="k", ls=":", label="paliers de la Figure 1")
    ax[0, 1].plot(t, [de.mission(x, de.MISSION_FIG1) for x in t], color="C3", label="consigne lissée")
    ax[1, 1].plot(t, [de.vitesse_mission(x, de.MISSION_FIG1) for x in t], color="C3")
    ax[0, 1].set_title("Mission finale : somme de sauts lissés (3 s et 7 s)")
    ax[0, 1].legend()
    for a in ax[1]:
        a.set_xlabel("t [s]")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_consigne.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 3. récompense
def fig_recompense():
    import env_z as de
    e = np.linspace(-1.5, 1.5, 601)
    fin = de.W_FIN * np.exp(-(e / de.SIG_FIN) ** 2)
    large = de.W_LARGE * np.exp(-(e / de.SIG_LARGE) ** 2)
    loin = -de.W_LOIN * np.minimum(np.abs(e), de.E_LOIN) / de.E_LOIN
    total = fin + large + loin
    # ancienne récompense (version 1) : -pseudo-Huber(e, 10 cm)
    d = 0.10
    huber = -(d ** 2) * (np.sqrt(1 + (e / d) ** 2) - 1)

    ev = np.linspace(-1.5, 1.5, 601)
    vit = -de.W_VIT * (1 - np.exp(-(ev / de.SIG_VIT) ** 2))
    du = np.linspace(-1, 1, 401)
    dou = -de.LAMBDA_DU * du ** 2

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].plot(e, fin, label="suivi fin (σ = 5 cm)")
    ax[0].plot(e, large, label="suivi large (σ = 30 cm)")
    ax[0].plot(e, loin, label="guidage lointain")
    ax[0].plot(e, total, "k", lw=2, label="somme (position)")
    ax[0].set_xlabel("écart de position e [m]")
    ax[0].set_ylabel("récompense par pas")
    ax[0].set_title("Termes de position (version 2)")
    ax[0].legend(fontsize=8)
    ax[1].plot(e, huber, color="C3", label="V1 : −pseudo-Huber (non borné)")
    ax[1].plot(e, total, "k", lw=2, label="V2 : somme des termes bornés")
    ax[1].set_ylim(-1.2, 1.2)
    ax[1].set_xlabel("écart de position e [m]")
    ax[1].set_title("Ancienne et nouvelle récompense (zoom)")
    ax[1].legend(fontsize=8)
    ax[2].plot(ev, vit, label="écart de vitesse e_v [m/s]")
    ax[2].plot(du, dou, label="variation de commande Δu")
    ax[2].set_xlabel("e_v ou Δu")
    ax[2].set_title("Amortissement et douceur")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_recompense.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 4. synthèse des contrôleurs
def fig_synthese():
    sac = json.load(open(VS / "agent_z_sac" / "results.json", encoding="utf8"))
    ppo = json.load(open(VS / "agent_z_ppo" / "results.json", encoding="utf8"))
    rows = {"LQI (gains ÷ 2)": ppo["lqi"], "SAC (run 1)": sac["agent_final"], "PPO (run 2)": ppo["agent_final"]}
    mets = [("rmse_lisse", "RMSE / consigne lissée [m]"), ("depassement", "dépassement [m]"),
            ("erreur_finale", "erreur finale [m]"), ("agitation_u", "agitation de u")]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.8))
    for a, (key, name) in zip(ax, mets):
        vals = [r[key] for r in rows.values()]
        bars = a.bar(list(rows.keys()), vals, color=["C1", "C0", "C2"])
        a.bar_label(bars, fmt="%.3f", fontsize=8)
        a.set_title(name, fontsize=10)
        a.tick_params(axis="x", labelrotation=20, labelsize=8)
        a.grid(axis="y", alpha=0.3)
    fig.suptitle("Comparaison sur 20 hauteurs cibles fixes, sans bruit (Python, modèle GRU)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_synthese.png", dpi=150)
    plt.close(fig)
    for k, r in rows.items():
        print(k, {m: round(r[m], 4) for m, _ in mets})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_donnees, fig_consigne, fig_recompense, fig_synthese):
        try:
            f()
            print("OK", f.__name__)
        except Exception as ex:  # noqa: BLE001
            print("ECHEC", f.__name__, repr(ex))
