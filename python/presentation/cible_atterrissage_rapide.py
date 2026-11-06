"""
CIBLE D'ATTERRISSAGE, VOITURES PLUS RAPIDES : taux de réussite de la mission finale sur 100 voitures (figure en anglais).

Même mission et même superviseur que cible_atterrissage.py (sim_mission.py), mêmes trois familles de contrôleurs
(presentation/commun.py), mesures bruitées (g_noise = 0,1), mais :
  - 100 voitures tirées au hasard (graines 1000 à 1099) au lieu de 13, pour donner un pourcentage de réussite ;
  - vitesses de 0,5 à 1,0 m/s au lieu de 0,25 à 0,7 m/s (même générateur que voiture.voiture_aleatoire, seules les
    plages de vitesse changent). 1 m/s reste dans le domaine d'entraînement de l'agent x (cibles jusqu'à 1 m/s).
Un vol réussit s'il se pose (z < 2 cm au-dessus de la plateforme, superviseur en phase 5) en moins de 60 s ; on
indique aussi s'il se pose sur la plateforme (20 x 16 cm) et à moins de 5 cm / 2 cm du centre.

    python python/presentation/cible_atterrissage_rapide.py     ->  rapport/presentation/06b_cible_atterrissage_rapide.png
                                                                    rapport/presentation/cible_atterrissage_rapide.json
    python python/presentation/cible_atterrissage_rapide.py --figure   # redessine seulement (à partir du .json)
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402

import commun as C  # noqa: E402
import cible_atterrissage as CA  # noqa: E402

N, GRAINE0, V_MIN, V_MAX, T_FIN = 100, 1000, 0.5, 1.0, 60.0
JSON = C.PRESENTATION / "cible_atterrissage_rapide.json"
PNG = C.PRESENTATION / "06b_cible_atterrissage_rapide.png"
PLATEFORME = (0.10, 0.08)          # demi-longueur, demi-largeur de la plateforme [m]
MARQ = {"RL": "o", "ARX": "s", "LQI": "^"}
NOMS_EN = {"RL": "RL agents (PPO)", "ARX": "LQ(I) on ARX model", "LQI": "double-integrator LQI"}


def voiture_rapide(rng):
    """voiture.voiture_aleatoire avec des vitesses de V_MIN à V_MAX (premier palier : V_MIN à 0,9 m/s)."""
    tv = np.sort(rng.uniform(0, 30, 4)); tk = np.sort(rng.uniform(6, 36, 4))
    vit = tuple((float(tv[i]), float(rng.uniform(2, 4)), float(rng.uniform(V_MIN, V_MAX))) for i in range(4))
    vit = ((0.0, float(rng.uniform(2, 4)), float(rng.uniform(V_MIN, 0.9))),) + vit[1:]
    cour = []
    for i in (0, 2):
        cour += [(float(tk[i]), 1.5, float(rng.choice([-1, 1]) * rng.uniform(1 / 8, 1 / 2.5))),
                 (float(max(tk[i + 1], tk[i] + 2)), 1.5, 0.0)]
    return dict(t_depart=float(rng.uniform(2, 6)), x0=float(rng.uniform(0.5, 2.5)), y0=float(rng.uniform(-1.5, 1.5)),
                cap0=float(rng.uniform(-np.pi, np.pi)), vitesse=vit, courbure=tuple(cour))


def main():
    res, voit = {f: [] for f in C.FAMILLES}, []
    for g in range(GRAINE0, GRAINE0 + N):
        T = C.table_voiture(voiture_rapide(np.random.default_rng(g)), T_FIN + 1)
        voit.append(dict(graine=g, v_max_ms=float(T["v"].max()), a_lat_max=float(np.abs(T["a_lat"]).max()),
                         a_long_max=float(np.abs(T["a_long"]).max())))
        for f in C.FAMILLES:
            R = C.vol_famille(f, T, T_FIN, graine=g)
            d, ph = CA.pose(R), R["y"][:, 4]
            e = dict(graine=g, phase_max=int(ph.max()))
            if d:
                i = int(np.argmax(ph >= 5))
                vv = R["voiture"]["v"][i] * np.array([np.cos(R["voiture"]["cap"][i]), np.sin(R["voiture"]["cap"][i])])
                e.update(d, v_rel_cms=float(100 * np.hypot(*(R["vrai"][i, 3:5] - vv))))
            # instant d'entrée dans chaque phase (0 stabilisation ... 5 posé)
            e["debuts_phases_s"] = [float(R["t"][np.argmax(ph >= k)]) if (ph >= k).any() else None for k in range(6)]
            res[f].append(e)
        print(f"voiture {g} : " + ", ".join(f"{f} {'posé' if 'avant_m' in res[f][-1] else 'ÉCHEC'}" for f in C.FAMILLES))
    out = dict(parametres=dict(n=N, graines=[GRAINE0, GRAINE0 + N - 1], vitesse_ms=[V_MIN, V_MAX], t_fin_s=T_FIN,
                               g_noise=0.1), voitures=voit, resultats=res, bilan=bilan(res))
    JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for f, b in out["bilan"].items():
        print(f, b)
    figure(out)


def bilan(res):
    B = {}
    for f in C.FAMILLES:
        P = [d for d in res[f] if "avant_m" in d]
        ec = np.array([100 * np.hypot(d["avant_m"], d["gauche_m"]) for d in P])
        sur = sum(abs(d["avant_m"]) < PLATEFORME[0] and abs(d["gauche_m"]) < PLATEFORME[1] for d in P)
        tp = np.array([d["t_pose_s"] for d in P])
        B[f] = dict(poses=len(P), sur_plateforme=int(sur), sous_5cm=int(np.sum(ec < 5)), sous_2cm=int(np.sum(ec < 2)),
                    ecart_moyen_cm=float(ec.mean()), ecart_median_cm=float(np.median(ec)), ecart_max_cm=float(ec.max()),
                    t_pose_moyen_s=float(tp.mean()), t_pose_min_s=float(tp.min()), t_pose_max_s=float(tp.max()),
                    v_rel_moyenne_cms=float(np.mean([d["v_rel_cms"] for d in P])),
                    v_rel_max_cms=float(np.max([d["v_rel_cms"] for d in P])))
    return B


def figure(out=None):
    C.style_clair()
    if out is None:
        out = json.loads(JSON.read_text(encoding="utf-8"))
    res, B, n = out["resultats"], out["bilan"], out["parametres"]["n"]
    fig = plt.figure(figsize=(13.5, 5.4))
    ax = fig.add_axes((0.045, 0.12, 0.31, 0.66))
    axb = fig.add_axes((0.45, 0.12, 0.22, 0.66))
    axt = fig.add_axes((0.80, 0.12, 0.185, 0.66))

    # (a) points de posé dans le repère de la voiture
    ax.add_patch(patches.Rectangle((-10, -8), 20, 16, fc="w", ec="k", lw=1.2, zorder=0))
    for r, ls in ((2, "-"), (5, "--")):
        ax.add_patch(patches.Circle((0, 0), r, fill=False, ec="0.4", ls=ls, lw=0.9, zorder=1))
        ax.text(r * 0.71 + 0.2, r * 0.71 + 0.2, f"{r} cm", fontsize=7.5, color="0.35")
    for f in C.FAMILLES:
        P = [d for d in res[f] if "avant_m" in d]
        ax.scatter([100 * d["avant_m"] for d in P], [100 * d["gauche_m"] for d in P], marker=MARQ[f], s=26,
                   c=C.COUL[f], ec="w", lw=0.5, alpha=0.85, zorder=3,
                   label=f"{NOMS_EN[f]}: mean {B[f]['ecart_moyen_cm']:.1f} cm, max {B[f]['ecart_max_cm']:.1f} cm")
    ax.set_xlim(-11, 13); ax.set_ylim(-9.5, 9.5); ax.set_aspect("equal")
    ax.annotate("", xy=(12.6, 0), xytext=(10.6, 0), arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.4))
    ax.text(11.6, 0.8, "car\nheading", fontsize=7, ha="center", color="0.3")
    ax.set_xlabel("forward [cm]"); ax.set_ylabel("left [cm]")
    ax.set_title("touchdown points on the platform (20 × 16 cm)", fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.27), fontsize=8, frameon=False)

    # (b) taux de réussite
    crit = (("landed < 60 s", "poses"), ("on the platform", "sur_plateforme"), ("within 5 cm", "sous_5cm"),
            ("within 2 cm", "sous_2cm"))
    y = np.arange(len(crit))
    for k, f in enumerate(C.FAMILLES):
        v = [100 * B[f][c] / n for _, c in crit]
        yy = y + (k - 1) * 0.26
        axb.barh(yy, v, 0.24, color=C.COUL[f], ec="w", lw=1)
        for yi, vi in zip(yy, v):
            axb.text(vi + 1, yi, f"{vi:.0f} %", va="center", fontsize=7.5, color="0.15")
            axb.plot(vi, yi, MARQ[f], color=C.COUL[f], ms=4.5, mec="w", mew=0.5)
    axb.set_yticks(y, [c for c, _ in crit], fontsize=9); axb.invert_yaxis()
    axb.set_xlim(0, 118); axb.set_xticks([0, 25, 50, 75, 100])
    axb.set_xlabel(f"share of the {n} cars [%]"); axb.grid(axis="x", alpha=0.35)
    axb.set_title("success rate", fontsize=10)

    # (c) instant du posé
    for k, f in enumerate(C.FAMILLES):
        tp = np.array([d["t_pose_s"] for d in res[f] if "t_pose_s" in d])
        jit = np.random.default_rng(k).uniform(-0.18, 0.18, len(tp))
        axt.scatter(tp, k + jit, marker=MARQ[f], s=14, c=C.COUL[f], alpha=0.55, ec="none")
        axt.plot([tp.mean()] * 2, [k - 0.32, k + 0.32], color="0.15", lw=2)
        axt.text(tp.mean(), k - 0.36, f"mean {tp.mean():.1f} s", ha="center", va="bottom", fontsize=8, color="0.15")
    axt.set_yticks(range(3), [NOMS_EN[f] for f in C.FAMILLES], fontsize=8.5); axt.invert_yaxis()
    axt.set_ylim(2.6, -0.75); axt.set_xlabel("touchdown time [s]"); axt.grid(axis="x", alpha=0.35)
    axt.set_title("when the drone lands", fontsize=10)

    p = out["parametres"]
    fig.suptitle(f"Final mission on {n} random cars at {p['vitesse_ms'][0]}–{p['vitesse_ms'][1]} m/s, turns of radius "
                 f"≥ 2.5 m, noisy measurements (g_noise = 0.1)", fontsize=12, y=0.985)
    fig.savefig(PNG, dpi=140); plt.close(fig)
    print(f"-> {PNG}")


if __name__ == "__main__":
    figure() if "--figure" in sys.argv else main()
